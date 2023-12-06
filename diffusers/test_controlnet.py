import argparse
import math
import os
import random
import shutil
import uuid
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

def predict_single(args, prompt, path_to_image, 
                    vae, text_encoder, tokenizer, unet, 
                    controlnet, device, weight_dtype, 
                    auxiliary_prompt = None):
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

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # predict 4 outputs for each input image
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(
                validation_prompt, validation_image, 
                num_inference_steps=args.num_inference_steps, 
                generator=generator,
                negative_prompt= validation_prompt_neg,
                guidance_scale = args.guidance_scale
            ).images[0]
            ## upscale the output back to origianl size
            # let's upscale it 4x by realesrgan first
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
            images.append(image)
    return images

def main(args):
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

    # output prediction
    path_to_img = Path(args.img)
    assert path_to_img.is_dir()

    dirs = ['left', 'right']

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

    for img in os.listdir(args.img):
        if 'png' not in img or 'color' in img: continue
        print("log:\topening %s"%img)
        for d in dirs:
            if 'line' in img:
                prompt, direction = gen_prompt_line(d)
                input_img_path = path_to_img / img
                flat = np.array(Image.open(path_to_img / img.replace('line', 'flat')).convert("RGB"))
                line = np.array(Image.open(path_to_img / img).convert("RGB")).mean(axis = -1).astype(float) / 255
            elif 'flat' in img:
                prompt, direction = gen_prompt_color(d)
                input_img_path = path_to_img / img.replace('flat', 'color')
                flat = np.array(Image.open(path_to_img / img).convert("RGB"))
                line = np.array(Image.open(path_to_img / img.replace('flat', 'line')).convert("RGB")).mean(axis = -1).astype(float) / 255
                if input_img_path.exists() is False:
                    Image.fromarray((flat * line).astype(np.uint8)).save(input_img_path)
            else:
                raise ValueError('not supported input %s!'%img)

            mask = flat.mean(axis = -1) == 255
            imgs = predict_single(args,
                [prompt, args.prompt_neg], input_img_path,
                vae, text_encoder, tokenizer, unet, controlnet, device, weight_dtype, args.prompt_aux)

            # extract shadow layer and save results
            out_path = Path('results')
            img_raw = Image.open(input_img_path)
            img_raw.save(out_path / img.replace('flat', 'color'))
            for i in range(len(imgs)):
                extract_shadow(imgs[i], img_raw, img.replace('flat', 'color'), direction, i, out_path, mask, line)

def extract_shadow(res, img, name, direction, idx, out_path, flat_mask, line = None):
    res.save(out_path/name.replace(".png", "_%s_res%d.png"%(direction, idx)))
    # shadow = (np.array(res).mean(axis = -1) < 127).astype(float)
    # shadow[flat_mask] = 1
    res_np = np.array(res).mean(axis = -1) / 255
    res_np[res_np >= 0.45] = 1
    if line is not None:
        line_mask = line < 0.5
        res_np[line_mask] = 1
    img_np = np.array(img)
    Image.fromarray((res_np*255).astype(np.uint8)).save(out_path/name.replace(".png", "_%s_shadow%d.png"%(direction, idx)))
    Image.fromarray((img_np * res_np[..., np.newaxis]).astype(np.uint8)).save(out_path/name.replace(".png", "_%s_blend%d.png"%(direction, idx)))
    

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="ShadowMagic SD backend v0.1")
    parser.add_argument(
        "img",
        type=str,
        help="Path to validation image",
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
        default="./pretrained/divineelegancemix_2x",
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
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())