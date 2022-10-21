# a small demo to show the shadow result
import torch
import argparse
import logging as log
import numpy as np
import cv2
import gradio as gr

from PIL import Image
from layers import UNet
from torchvision.transforms import functional as F

def resize_hw(h, w, size):
    # we resize the shorter edge to the target size
    if h > w:
        ratio =  h / w
        h = int(size * ratio)
        w = size
    else:
        ratio = w / h
        w = int(size * ratio)
        h = size
    return h, w

def remove_alpha(img, gray = False):
    if len(img.shape) == 3:
        h, w, c = img.shape
        if c == 4:
            alpha = np.expand_dims(img[:, :, 3], -1) / 255
            whit_bg = np.ones((h, w, 3)) * 255
            img_res = img[:, :, :3] * alpha + whit_bg * (1 - alpha)
            if gray:
                img_res = img_res.mean(axis = -1)
        else:
            img_res = img
    else:
        img_res = img
    return img_res

def init_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info('Using device %s'%device)
    net = UNet(in_channels = 3, out_channels = 1, bilinear = True)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    log.info('Model loaded from %s'%model_path)
    return net

def init_from_args(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DIR_TO_FLOAT = {"right": 0.25, "left":0.5, "back":0.75, "top":1.0}
    # read input
    direction = DIR_TO_FLOAT.get(args.d, None)
    if direction is None:
        log.ERROR(f'Unsupport direction {args.d}')
        raise ValueError("unsupported light direction!")
    label = torch.Tensor([direction]).to(device).unsqueeze(0).float()
    line = np.array(Image.open(args.l))
    line = 255 - line[:, :, 3]
    flat = np.array(Image.open(args.l.replace("line", "flat")))
    shadow = np.array(Image.open(args.l.replace("line", "shadow")))
    flat = remove_alpha(flat)
    img_np = flat * (np.expand_dims(line, axis = -1) / 255)
    h, w = line.shape[0], line.shape[1]
    h, w = resize_hw(h, w, args.s)
    img_np = cv2.resize(img_np, (w, h), interpolation = cv2.INTER_AREA)
    shadow = cv2.resize(shadow, (w, h), interpolation = cv2.INTER_NEAREST)
    shadow = norm_shadow(shadow)
    img = F.to_tensor(img_np / 255).to(device).unsqueeze(0).float()
    img = F.normalize(img, 0.5, 0.5)
    return img, img_np, shadow, label

def shadowing(net, img, label):
    net.eval()
    with torch.no_grad():
        pre, _, _, _ = net(img, label)
        pre = torch.sigmoid(pre)
        pre = pre.detach().cpu().numpy().squeeze()
    return pre

def norm_shadow(shadow):
    shadow = (shadow) / 255
    shadow[np.where(shadow == 0)] = 0.5
    return shadow

def overlay_shadow(img, shadow):
    return (img * np.expand_dims(shadow, axis = -1)).astype(np.uint8) 

def thres_norm_shadowmap(pre, thres = 0.5):
    pre = (pre <= 0.5).astype(float)
    pre[np.where(pre == 0)] = 0.5
    return pre

def lauch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("Please upload the image which need to be shadowed")
        with gr.Row():
            with gr.Column(scale = 1):
                img_input = gr.Image()
                run_button = gr.Button("Get Shadow")
            with gr.Column(scale = 1):
                img_shadowmap = gr.Image()
                img_res = gr.Image()
                shadow_thres = gr.Slider(minimum = 0, maximum = 1, value = 0.5, step = 0.01, label = "Shadow level")
                adjust_thres_button = gr.Button("Re-shading")

def main():
    ## overall init
    args = parse()
    log.basicConfig(level = log.INFO, format='%(levelname)s:%(message)s')
    # init model  
    net = init_model(args.m)

    ## run in command line mode or WebUI mode
    if args.g:
        pass
    else:
        img, img_np, shadow, label = init_from_args(args)
        pre = shadowing(net, img, label)
        pre = thres_norm_shadowmap(pre) # threshold is 0.5 by default
        img_pred = overlay_shadow(img_np, pre)
        img_gt = overlay_shadow(img_np, shadow)
        img = np.concatenate((img_pred, img_gt), axis = 1)    
        Image.fromarray(img.astype(np.uint8)).save(args.l.replace("line", "res"))

def parse():
    parser = argparse.ArgumentParser(description = "ShadowMagic Demo Ver 0.1",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', type = str, default = './checkpoints/test/AP_GFMask1.pth',
        help = 'the path to the pretrained model')
    parser.add_argument('-l', type = str, 
        help = 'the path to input line drawing')
    parser.add_argument('-g', action = 'store_true', 
        help = 'enable gradio for a interactive UI')
    parser.add_argument('-d', type = str, 
    	help = 'light direction, should be one of the following: right, left, back, top')
    parser.add_argument('-s', type = int, default = 1024,
    	help = 'the target size for shrinking the input image if it is too large')
    return parser.parse_args()

if __name__ == "__main__":
    main()