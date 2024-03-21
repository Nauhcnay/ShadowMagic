import argparse
import math
import os
import random
import shutil
import uuid
import cv2
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from post_process import shadow_refine_2nd, gkern

import sys 
sys.path.append("../wgan/")
from utils.preprocess import flat_to_fillmap, fillmap_to_color

check_min_version("0.22.0.dev0")

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def predict_single(args, prompt, path_to_image, vae, text_encoder, tokenizer, unet, controlnet, device, weight_dtype, auxiliary_prompt = None):
    path_to_realesrgan = Path("Real-ESRGAN")
    if path_to_realesrgan.exists() == False:
        os.system("git clone https://github.com/xinntao/Real-ESRGAN")
        os.system('''
            cd Real-ESRGAN
            pip install basicsr
            pip install facexlib
            pip install gfpgan
            pip install -r requirements.txt
            python setup.py develop'''
            )
    pyfile = "inference_realesrgan.py"
    assert (path_to_realesrgan/pyfile).exists()
    validation_prompt, validation_prompt_neg = prompt
    assert isinstance(validation_prompt, str)
    if auxiliary_prompt is not None:
        assert isinstance(auxiliary_prompt, str)
        validation_prompt = validation_prompt + ", " + auxiliary_prompt

    raw_img = Image.open(path_to_image).convert("RGB")
    w, h = raw_img.size
    # downsize if necessary
    if h > 1024 or w > 1024:
        long_side = h if h > w else w
        ratio = 1024 / long_side
        h_new = int(h * ratio)
        w_new = int(w * ratio)
        validation_image = raw_img.resize((w_new, h_new))
    elif 'shadesketch' in str(path_to_image):
        validation_image = real_esrgan_resize(raw_img, 1024, 1024, path_to_realesrgan, pyfile)
    else:
        validation_image = raw_img

    # https://huggingface.co/docs/diffusers/v0.24.0/en/api/pipelines/overview#diffusers.DiffusionPipeline
    # more parameter explaination
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    # predict 4 outputs for each input image
    images = []
    seeds = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            generator = torch.Generator(device=device)
            if args.seed is None:
                generator = None
            elif args.seed == -1:
                seed = generator.seed()
                generator = generator.manual_seed(seed)
            else:
                generator = generator.manual_seed(args.seed)
            image = pipeline(
                validation_prompt, validation_image, 
                num_inference_steps=args.num_inference_steps, 
                generator=generator,
                negative_prompt= validation_prompt_neg,
                guidance_scale = args.guidance_scale
            ).images[0]
            if 'shadesketch' not in str(path_to_image):
                ## upscale the output back to origianl size
                # let's upscale it 4x by realesrgan first
                image = real_esrgan_resize(image, h, w, path_to_realesrgan, pyfile)
            images.append(image)
            if args.seed == -1:
                seeds.append(seed)
            else:
                seeds.append(args.seed)
    return images, seeds

def real_esrgan_resize(image, h, w, path_to_realesrgan, pyfile):
    temp_png = str(uuid.uuid4()) + ".png"
    out_path = Path(temp_png.replace(".png", ""))
    png_4x = temp_png.replace(".png", "_out.png")
    image.save(temp_png)
    cmd  = "python %s -n RealESRGAN_x4plus_anime_6B -i %s -o %s -s 2"%(path_to_realesrgan/pyfile, temp_png, out_path)
    os.system(cmd)
    image = Image.open(out_path/png_4x)
    # resize back to original size
    image = image.resize((w, h))
    # clean up
    os.remove(temp_png)
    shutil.rmtree(out_path)
    return image

def init_controlnet(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # load pretrained control net
    print('log:\tLoading existing controlnet weights')

    # models will be set to evaluation mode by default
    # https://huggingface.co/docs/diffusers/v0.24.0/en/api/models/overview#diffusers.ModelMixin.from_pretrained
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    return vae, text_encoder, tokenizer, unet, controlnet, device, weight_dtype

def gen_prompt_line(dirs):
    if isinstance(dirs, str):
        direction = dirs
    else:    
        direction = random.sample(dirs, k = 1)[0]
    return "add shadow from %s lighting"%direction, direction

def gen_prompt_color(dirs):
    if isinstance(dirs, str):
        direction = dirs
    else:    
        direction = random.sample(dirs, k = 1)[0]
    return "add shadow from %s lighting and remove color"%direction, direction

def predict_and_extract_shadow( 
    path_to_img, 
    img,
    vae, 
    text_encoder, 
    tokenizer, 
    unet, 
    controlnet, 
    device, 
    weight_dtype,
    args,
    direction = 'left',
    to_png = True):
    print("log:\topening %s"%img)
    if 'flat' in img:
        prompt, direction = gen_prompt_color(direction)
        input_img_path = path_to_img / img.replace('flat', 'color')
        flat = np.array(Image.open(path_to_img / img))
        if flat.shape[-1] == 4:
            bg = np.ones((flat.shape[0], flat.shape[1], 3)) * 255
            alpha = flat[..., -1][..., np.newaxis] / 255
            rgb = flat[..., 0:3]
            flat = rgb * alpha + bg * (1 - alpha)
            Image.fromarray(flat.astype(np.uint8)).save(path_to_img / img)
        line = np.array(Image.open(path_to_img / img.replace('flat', 'line')))
        if line.shape[-1] == 4:
            line = 255 - line[..., -1]
            Image.fromarray(line.astype(np.uint8)).save(path_to_img / img.replace('flat', 'line'))
            line = line.astype(float) / 255
        elif len(line.shape) == 2:
            line = line / 255
        else:
            line = line.mean(axis = -1).astype(float) / 255
        if input_img_path.exists() is False:
            Image.fromarray((flat * line[..., np.newaxis]).astype(np.uint8)).save(input_img_path)
    elif 'shadesketch' in img:
        prompt, direction = gen_prompt_color(direction)
        input_img_path = path_to_img / img

    else:
        raise ValueError('not supported input %s!'%img)

    
    imgs, seeds = predict_single(args,
        [prompt, args.prompt_neg], input_img_path,
        vae, text_encoder, tokenizer, unet, controlnet, device, weight_dtype, args.prompt_aux)

    # extract shadow layer and save results
    out_path = Path('results')
    img_raw = Image.open(input_img_path)
    img_raw.save(out_path / img.replace('flat', 'color'))
    shadows = []
    for i in range(len(imgs)):
        shadows.append(extract_shadow(
            imgs[i], 
            img_raw, 
            img.replace('flat', 'color'), 
            direction, 
            i, 
            out_path,
            flat, 
            line,
            seeds[i],
            to_png = to_png))
    return shadows

def extract_shadow(res, img, name, direction, idx, out_path, flat, line = None, seed = None, to_png = True):
    flat_mask = flat.mean(axis = -1) == 255
    res_np = (np.array(res).mean(axis = -1) / 255) < 0.65
    res_np[flat_mask] = False
    
    print("log:\trefine predicted shadow")
    # convert flat to fill
    fill, _ = flat_to_fillmap(flat, False)
    # aa = fillmap_to_color(fill)
    # Image.fromarray(res_np).save("aa.png")
    res_np = shadow_refine_2nd(fill, res_np, line < 0.5)
    # Image.fromarray(res_np).save("bb.png")

    # convert shadow flag map into shadows
    res_np_overlay = res_np.astype(float).copy()
    res_np_overlay[res_np == 1] = 0.5
    res_np_overlay[res_np == 0] = 1
    res_np = ~res_np

    img_np = np.array(img)
    
    if to_png:
        if seed is None:
            Image.fromarray((img_np * res_np_overlay[..., np.newaxis]).astype(np.uint8)).save(out_path/name.replace(".png", "_%s_blend%d.png"%(direction, idx)))
        else:
            Image.fromarray((img_np * res_np_overlay[..., np.newaxis]).astype(np.uint8)).save(out_path/name.replace(".png", "_%s_%d_blend%d.png"%(direction, seed, idx)))
        
        if seed is None:
            Image.fromarray((res_np*255).astype(np.uint8)).save(out_path/name.replace(".png", "_%s_shadow%d.png"%(direction, idx)))
        else:
            Image.fromarray((res_np*255).astype(np.uint8)).save(out_path/name.replace(".png", "_%s_%d_shadow%d.png"%(direction, seed, idx)))
    # res_np = (res_np*255).astype(np.uint8)
    return res_np

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="ShadowMagic SD backend v0.1")
    parser.add_argument(
        "--img",
        type=str,
        help="Path to validation image",
        default = None,
        required = False
    )
    parser.add_argument(
        "--prompt_neg",
        type=str,
        default=None,
        required=False,
        help="Optional negative prompt that might enhance the generation result",
    )
    parser.add_argument(
        "--prompt_aux",
        type=str,
        default=None,
        required=False,
        help="Optional auxiliary positive prompt that might enhance the generation result",
    )

    # let's try anything v5 frist and see if that works good
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stablediffusionapi/divineelegancemix",
        # default="./checkpoints/divineelegancemix_2x/checkpoint-14000/",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default="./checkpoints/divineelegancemix_2x/checkpoint-14000/controlnet",
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training.")
    parser.add_argument(
        "--port_to_frontend", 
        type=int, 
        default=8000, 
        help="A seed for reproducible training.")

    args = parser.parse_args()
    return args

def run_single(user, flat, line, color, name, direction = 'left'):
    # set up running folder
    temp_folder = Path("temp_per_user")
    temp_folder = temp_folder / user
    if os.path.exists(temp_folder) == False:
        os.makedirs(temp_folder)
    name_flat = name + '_flat.png'
    # for debug
    Image.fromarray(flat).save(temp_folder/(name+'_flat.png'))
    Image.fromarray(line).save(temp_folder/(name+'_line.png'))
    Image.fromarray(color).save(temp_folder/(name+'_color.png'))
    
    path_to_img = temp_folder
    assert path_to_img.is_dir()
    img = name_flat

    # init network
    args = parse_args()
    vae, text_encoder, tokenizer, unet, controlnet, device, weight_dtype = init_controlnet(args)

    # predict
    shadows = predict_and_extract_shadow(
        path_to_img, 
        img,
        vae, 
        text_encoder, 
        tokenizer, 
        unet, 
        controlnet, 
        device, 
        weight_dtype,
        args,
        direction,
        to_png = False)
    return shadows

def main(args):
    # wrap all init codes into one function
    vae, text_encoder, tokenizer, unet, controlnet, device, weight_dtype = init_controlnet(args)
    
    # output prediction
    assert args.img is not None
    path_to_img = Path(args.img)
    assert path_to_img.is_dir()

    # dirs = ['left', 'right', "top", "back"]
    dirs = ['left', 'right']

    for img in os.listdir(args.img):
        if 'png' not in img or 'color' in img or 'line' in img: continue
        for d in dirs:
            predict_and_extract_shadow(
                path_to_img, 
                img,
                vae, 
                text_encoder, 
                tokenizer, 
                unet, 
                controlnet, 
                device, 
                weight_dtype,
                args,
                direction = d)

if __name__ == '__main__':
    main(parse_args())