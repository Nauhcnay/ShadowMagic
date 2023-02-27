# a small demo to show the shadow result
import torch
import argparse
import logging as log
import numpy as np
import cv2
import gradio as gr
import os

from PIL import Image
from layers import UNet
from torchvision.transforms import functional as F
from functools import partial
from os.path import join, exists
from scipy.signal import convolve2d as conv2d
from utils.l0_gradient_minimization import l0_gradient_minimization_2d as l0_2d
from utils.misc import resize_hw, remove_alpha
from sklearn.cluster import DBSCAN
from skimage.segmentation import felzenszwalb, mark_boundaries


device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIR_TO_FLOAT = {"right": 0.25, "left":0.5, "back":0.75, "top":1.0}
net = None

def init_model(model_path):
    log.info('Using device %s'%device)
    global net
    net = UNet(in_channels = 3, out_channels = 1, bilinear = True)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    log.info('Model loaded from %s'%model_path)

def init_from_args(line, size, direction = None):
    # read input
    if direction is None:
        direction = line.split("_")[-2]
    direction = DIR_TO_FLOAT.get(direction, None)
    if direction is None:
        log.ERROR(f'Unsupport direction {direction}')
        raise ValueError("unsupported light direction!")
    label = torch.Tensor([direction]).to(device).unsqueeze(0).float()
    line_np = np.array(Image.open(line))
    line_np = 255 - line_np[:, :, 3]
    flat = np.array(Image.open(line.replace("line", "flat")))
    shadow = np.array(Image.open(line.replace("line", "shadow")))
    flat = remove_alpha(flat)
    img_np = flat * (np.expand_dims(line_np, axis = -1) / 255)
    h, w = line_np.shape[0], line_np.shape[1]
    h, w = resize_hw(h, w, size)
    img_np = cv2.resize(img_np, (w, h), interpolation = cv2.INTER_AREA)
    shadow = cv2.resize(shadow, (w, h), interpolation = cv2.INTER_NEAREST)
    shadow = norm_shadow(shadow)
    img = F.to_tensor(img_np / 255).to(device).unsqueeze(0).float()
    img = F.normalize(img, 0.5, 0.5)
    return img, img_np, shadow, label

def shadowing(img, label):
    net.eval()
    with torch.no_grad():
        pre, _, _, _ = net(img, label)
        pre = torch.sigmoid(pre)
        pre = pre.detach().cpu().numpy().squeeze()
    torch.cuda.empty_cache()
    return pre

def norm_shadow(shadow):
    shadow = (shadow) / 255
    shadow[np.where(shadow == 0)] = 0.5
    return shadow

def overlay_shadow(img, shadow):
    if len(shadow.shape) == 3:
        shadow = shadow.mean(axis = -1)
    return (img * np.expand_dims(shadow, axis = -1)).astype(np.uint8) 

def thres_norm_shadowmap(pre, thres = 0.5):
    pre = (pre <= thres).astype(float)
    pre[np.where(pre == 0)] = 0.5
    return pre

def lauch_gradio():
    examples = read_examples()
    with gr.Blocks() as demo:
        gr.Markdown("Please upload the image which need to be shadowed")
        # draw the UI componements
        with gr.Row():
            with gr.Column(scale = 1):
                img_input = gr.Image(label = "Drawing")
                shadow = gr.Image(label = "Ground Truth Shadow (optional)")
                direction = gr.Radio(choices = ["left", "right", "back", "top"], value = "right", label = "Light Direction")
                resize = gr.Radio(choices = ["512", "1024"], label = "Resize")
                run_button = gr.Button("Get Shadow")
            with gr.Column(scale = 1):
                img_pre = gr.Image()
                img_heatmap = gr.Image()
                shadow_thres = gr.Slider(minimum = 0, maximum = 1, value = 0.5, step = 0.01, label = "Shadow Level")
                adjust_thres_button = gr.Button("Re-shading")
        # write the botton function
        run_button.click(pre_gradio, inputs = [img_input, shadow, direction, resize, shadow_thres], outputs = [img_pre, img_heatmap])
        adjust_thres_button.click(reshading, inputs = [img_input, img_heatmap, shadow_thres, resize], outputs = img_pre)
        gr.Examples(examples = examples, inputs = [img_input, direction, resize, shadow, shadow_thres])
    demo.launch(share = False)

def pre_gradio(img_np, shadow, label, resize, shadow_thres):
    label = torch.Tensor([DIR_TO_FLOAT[label]]).to(device).unsqueeze(0).float()
    h, w = img_np.shape[0], img_np.shape[1]
    h, w = resize_hw(h, w, int(resize))
    img_np = cv2.resize(img_np, (w, h), interpolation = cv2.INTER_AREA)
    img = img = F.to_tensor(img_np / 255).to(device).unsqueeze(0).float()
    img = F.normalize(img, 0.5, 0.5)
    pre = shadowing(img, label)
    pre_default = thres_norm_shadowmap(pre, shadow_thres)
    img_pre = overlay_shadow(img_np, pre_default)
    if shadow is not None:
        shadow = norm_shadow(shadow.mean(axis = -1))
        shadow = cv2.resize(shadow, (w, h), interpolation = cv2.INTER_NEAREST)
        img_gt = overlay_shadow(img_np, shadow)
        img_pre = np.concatenate((img_pre, img_gt), axis = 1)
    return img_pre.astype(np.uint8), (pre * 255).astype(np.uint8)

def read_examples():
    log.info("Loading examples")
    sample_path = './samples/'
    examples = []
    for p in os.listdir(sample_path):
        if "line" in p:
            label = p.split('_')[1]
            resize = "1024"
            line = np.array(Image.open(join(sample_path, p)))
            line = 255 - line[:, :, 3]
            flat = np.array(Image.open(join(sample_path, p.replace('line', 'flat'))))
            flat = remove_alpha(flat)
            img_np = flat * (np.expand_dims(line, axis = -1) / 255)
            drawing_path = join(sample_path, p.replace('line', 'drawing'))
            if exists(drawing_path) == False:
                Image.fromarray(img_np.astype(np.uint8)).save(drawing_path)
            shadow_path = join(sample_path, p.replace('line', 'shadow'))
            examples.append([drawing_path, label, resize, shadow_path, 0.5])
    return examples

def reshading(img_np, heatmap, thres, resize):
    h, w = img_np.shape[0], img_np.shape[1]
    h, w = resize_hw(h, w, int(resize))
    # import pdb
    # pdb.set_trace()
    heatmap = heatmap.astype(float) / 255
    img_np = cv2.resize(img_np, (w, h), interpolation = cv2.INTER_AREA)
    shadow = thres_norm_shadowmap(heatmap, thres)
    img_pre = overlay_shadow(img_np, shadow)
    return img_pre.astype(np.uint8)

def vis_heatmap(pre):
    pre_mask = pre > 125
    pre = pre - 125
    pre[~pre_mask] = 0
    color_nums = len(np.unique(pre))
    colors = (np.random.rand(color_nums + 1, 3) * 255).astype(np.uint8)
    colors[0] = np.array([0, 0, 0], dtype = np.uint8)
    return colors[pre]

def main():
    ## overall init
    args = parse()
    log.basicConfig(level = log.INFO, format='%(levelname)s:%(message)s')
    # init model  
    init_model(args.m)
    def thres(img, t = 150):
        img[img > t] = 255
        img[img <= t] = 0
        return 255 - img

    ## run in command line mode or WebUI mode
    if args.g:
        lauch_gradio()
    else:
        for line in os.listdir(args.l):
            if 'line' not in line: continue
            if os.path.exists(line.replace("line", "res")): continue
            line = join(args.l, line)
            img, img_np, shadow, label = init_from_args(line, args.s, args.d)
            pre = shadowing(img, label)
            pre_ = (pre.copy() * 255).astype(np.uint8)
            pre_[pre_ < 100] = 0
            pre_seg_label = felzenszwalb(pre_, scale = 32, sigma = 0.5, min_size = 32)
            pre_seg = mark_boundaries(pre_, pre_seg_label)
            Image.fromarray((pre_seg * 255).astype(np.uint8)).show()

            # I think filtering seems not work
            # add gaussain filter
            pre_g = cv2.GaussianBlur(pre_, (5, 5), 0) 
            # add billateral filter
            pre_b = cv2.bilateralFilter(pre_, 15, 150, 75)
            # add L0 gradient minimization filter
            pre_l = l0_2d(pre_, lmd = 0.005, beta_max = 1.0e5, beta_rate = 1.5, max_iter = 100).astype(np.uint8)
            shadow_np = np.repeat((shadow * 255).astype(np.uint8)[..., np.newaxis], 3, axis = -1)
            pre_np = np.repeat((pre * 255).astype(np.uint8)[..., np.newaxis], 3, axis = -1)
            pre_g_np = np.repeat(pre_g.astype(np.uint8)[..., np.newaxis], 3, axis = -1)
            pre_b_np = np.repeat(pre_b.astype(np.uint8)[..., np.newaxis], 3, axis = -1)
            # pre_l_np = vis_heatmap(pre_l)
            pre_l_np = np.repeat(pre_l.astype(np.uint8)[..., np.newaxis], 3, axis = -1)
            pres = np.concatenate((shadow_np, 255 - pre_np, 255 - pre_g_np, 255 - pre_b_np, 255 - pre_l_np), axis = 1)
            pres_thres = np.concatenate((thres(shadow_np), thres(255 - pre_np), thres(255 - pre_g_np), thres(255 - pre_b_np), thres(255 - pre_l_np)), axis = 1)
            Image.fromarray(pres).save(line.replace('line', "pres"))
            Image.fromarray(pres_thres).save(line.replace('line', "pres_thres"))
            
            # let's try clustering


            pre = thres_norm_shadowmap(pre) # threshold is 0.5 by default
            img_pre = overlay_shadow(img_np, pre)
            img_gt = overlay_shadow(img_np, shadow)
            Image.fromarray(img_pre.astype(np.uint8)).save(line.replace('line', "pre_all"))
            Image.fromarray(img_gt.astype(np.uint8)).save(line.replace('line', "gt_all"))
            # Image.fromarray((pre*255).clip(0, 255).astype(np.uint8)).save(line.replace('line', "heatmap"))
            
            # visualize the true positive (green), false positive (red), ture negative (black), false negative (blue) regions of the prediction
            # pre_ = pre > shadow_thres
            # shadow = np.logical_not(shadow == 1)
            # h, w = shadow.shape
            # res = np.zeros((h, w))
            # mask_tp = np.logical_and(pre_, shadow)
            # mask_fp = np.logical_and(pre_, np.logical_not(shadow))
            # mask_tn = np.logical_and(np.logical_not(pre_), np.logical_not(shadow))
            # mask_fn = np.logical_and(np.logical_not(pre_), shadow)
            # d = [255, 255, 255]
            # r = [255, 0, 0]
            # g = [0, 0, 0]
            # b = [0, 0, 255]
            # colors = np.array([d, r, g, b])
            # res[mask_tn] = 0
            # res[mask_fp] = 1
            # res[mask_tp] = 2
            # res[mask_fn] = 3
            # res = colors[res.astype(int)]
            # Image.fromarray(res.astype(np.uint8)).save(line.replace('line', "comp"))

# thanks for https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)      


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
    	help = 'light direction, should be one of the following: right, left, back, top',
        default = None)
    parser.add_argument('-s', type = int, default = 1024,
    	help = 'the target size for shrinking the input image if it is too large')
    return parser.parse_args()

if __name__ == "__main__":
    main()