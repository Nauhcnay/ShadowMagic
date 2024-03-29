# refine the shadow and make it controlable 
import os
import cv2
import sys
sys.path.append("../wgan/")
import numpy as np

from PIL import Image
from scipy.signal import convolve2d as conv
from utils.preprocess import flat_to_fillmap, fillmap_to_color, flat_refine
from os.path import join
from skimage.morphology import skeletonize

K = np.array([
        [0,1,0], 
        [1,1,1], 
        [0,1,0], 
        ]).astype(float)

def display_img(img, shadow, line, flat, resize_to = 1024):
    # read images
    if isinstance(img, str):
        img = cv2.imread(img)
    if isinstance(shadow, str):
        shadow = cv2.imread(shadow)
    if isinstance(line, str):
        line = cv2.imread(line)
    if isinstance(flat, str):
        flat = cv2.imread(flat)
        flat, _ = flat_to_fillmap(flat)
    if len(shadow.shape) > 2:
        shadow = shadow.mean(axis = -1)
    # turn line into a mask which indicates the shadow boundary
    if len(line.shape) > 2:
        if line.shape[-1] == 4:
            line = 255 - line[..., 3]
            line = line < 255
        else:
            line = (line[..., 0] < 255)
    # turn the shadow to binary
    shadow = to_binary_shadow(shadow)
    h, w  = img.shape[0], img.shape[1]
    long_side = h if h > w else w
    if long_side > resize_to:
        ratio = resize_to / long_side
    else:
        ratio = 1
    h_new = int(h * ratio)
    w_new = int(w * ratio)
    if ratio != 1:
        img = cv2.resize(img, (w_new, h_new), interpolation = cv2.INTER_AREA)
        shadow = cv2.resize(shadow.astype(np.uint8), (w_new, h_new), interpolation = cv2.INTER_NEAREST)
        flat = cv2.resize(flat.astype(np.uint8), (w_new, h_new), interpolation = cv2.INTER_NEAREST)
        line = cv2.resize(line.astype(np.uint8), (w_new, h_new), interpolation = cv2.INTER_NEAREST).astype(float)

    # shadow = shadow_refine_2nd(flat, shadow, line)
    bg_mask_decrease = ~(shadow.astype(bool))
    bg_mask_increase = flat == 0
    # add boundary to line
    line = line.astype(bool) & shadow.astype(bool)
    line_skel = skeletonize(line.astype(bool))
    line_skel[0,:] = True
    line_skel[-1,:] = True
    line_skel[:, 0] = True
    line_skel[:, -1] = True

    # shadow_fill, _ = fillmap_to_color(shadow_fill)
    while True:
        res = (img * to_blend_shadow(shadow)[..., np.newaxis]).astype(np.uint8)
        cv2.imshow('results',res)
        k = cv2.waitKey(100)
        if k == ord('a'):
            shadow = decrease_shadow_gaussian(shadow, line_skel, bg_mask_decrease)
        elif k == ord('d'):
            shadow = increase_shadow_gaussian(shadow, line, bg_mask_increase)
        elif k == ord('q'):
            break
    cv2.destroyAllWindows()

def shadow_refine_2nd(fill, shadow, line):
    h, w = shadow.shape[0], shadow.shape[1]
    long_side = h if h > w else w
    shadow = shadow.astype(bool)
    # remove all bleeding shadow regions
    shadow_fill = np.zeros(shadow.shape).astype(int)
    fill_num = 1
    for r in np.unique(fill):
        if r == -1: continue   
        mask_per_flat = fill == r
        ss = shadow.copy()
        ss[~mask_per_flat] = False
        _, shadows = cv2.connectedComponents(ss.astype(np.uint8), connectivity = 4)
        for sr in np.unique(shadows):
            if sr == 0: continue
            mask_per_shadow = shadows == sr
            mask_per_shadow = (~(mask_per_shadow & line)) & mask_per_shadow 
            shadow_fill[mask_per_shadow] = fill_num
            fill_num += 1
            # this makes the function extremely slow!
            # mask_per_shadow_skel = skeletonize(mask_per_shadow)
            # if mask_per_shadow.sum() / mask_per_shadow_skel.sum() < 10:
            if mask_per_shadow.sum() < (300 * (long_side / 1024)):
                shadow[mask_per_shadow] = False
    
    # fill all small holes
    shadow_be_filled = ~(shadow.astype(bool) | (line == 1))
    shadow_fill = np.zeros(shadow.shape).astype(int)
    fill_num = 1
    _, holes = cv2.connectedComponents(shadow_be_filled.astype(np.uint8), connectivity = 4)
    # img_holes, _ = fillmap_to_color(holes)
    
    for h in np.unique(holes):
        if h == 0: continue
        mask_per_hole = holes == h
        shadow_fill[mask_per_hole] = fill_num
        fill_num += 1
        if mask_per_hole.sum() < (300 * (long_side / 1024)):
            shadow[mask_per_hole] = True
    shadow = shadow | (line == 1)
    return shadow

def increase_shadow_gaussian(shadow, line, bg_mask):
    # this function doesn't work still...
    K = gkern()
    shadow = shadow.astype(bool)
    shadow_conv = shadow.copy().astype(bool)
    shadow_conv[line.astype(bool)] = -0.9
    shadow_conv = conv(shadow_conv, K, mode='same')
    shadow = shadow | (shadow_conv > 0.8)
    shadow[bg_mask] = False
    return shadow

def decrease_shadow_gaussian(shadow, line, bg_mask, iters = 3):
    # gaussian blur based shadow decreasing
    K = gkern()
    shadow_conv = shadow.astype(float)
    for _ in range(iters): 
        shadow_conv[line.astype(bool)] = 0.9 * iters
        shadow_conv = conv(shadow_conv, K, mode='same')
    shadow_conv[bg_mask] = 0
    shadow_conv[shadow_conv < 0.5] = 0
    
    # add this blur to make the shadow edge smooth
    return conv(shadow_conv, K, mode='same')

# def decrease_shadow_erosion(shadow, line):
#     # erosion based shadow decreasing
#     shadow = shadow.copy()
#     removed = np.zeros(shadow.shape).astype(bool)
#     shadow_conv = shadow.astype(float)
#     shadow_conv[line.astype(bool)] = 255
#     shadow_conv = conv(shadow_conv, K, mode='same')
#     removed[(shadow_conv > 0) & (shadow_conv < 5)] = True
#     shadow[removed] = False
#     return shadow

# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def to_blend_shadow(shadow, thr = 0.9, smoothing = True, for_psd = False):
    res = np.zeros(shadow.shape)
    # binarize the shadow
    shadow = (shadow >= thr).astype(float)
    if smoothing:
        ## add post process as Ray's suggestion
        shadow_conv = conv(shadow.astype(np.uint8), K, mode = 'same')
        res[shadow >= thr] = 0.5
        res[shadow < thr] = 1
        res[shadow_conv == 2] = 0.75
    else:
        res[shadow >= thr] = 0.5
        res[shadow < thr] = 1
    # convert to array with transparency
    if for_psd:
        res_ = np.ones((res.shape[0], res.shape[1], 4)) * 255
        res_[..., 3] = res * 255
        res = res_
    return res

def to_binary_shadow(shadow):
    return shadow != 255  

def decrease_shadow(shadow, line):
    # preprocess line map 
    line = skeletonize(line.astype(bool))
    line[0,:] = 1
    line[-1,:] = 1
    line[:, 0] = 1
    line[:, -1] = 1
    bg_mask = ~shadow
    # decrease shadow 
    shadow = decrease_shadow_gaussian(shadow, line, bg_mask, iters = 3)
    return to_blend_shadow(shadow, for_psd =  True)

if __name__ == '__main__':
    # refine all flat regions in the results folder
    for img in os.listdir('./results/'):
        if 'flat' not in img or 'png' not in img: continue
        print('log:\topening %s'%img)
        flat = np.array(Image.open(join('./results/', img)))
        line = np.array(Image.open(join('./results/', img.replace('flat', 'line')))).mean(axis = -1)
        flat_refined, fill = flat_refine(flat, line)
        # np.save(join('./results/', img.replace('.png', '.npy')), fill)
        Image.fromarray(flat_refined).save(join('./results/', img))

    # display_img(
    #     "./ShadowMagic_output_shadows/image004_color.png", 
    #     "./ShadowMagic_output_shadows/image004_color_left_shadow0.png", 
    #     "./ShadowMagic_output_shadows/image004_line.png", 
    #     "./ShadowMagic_output_shadows/image004_flat.png")

    # display_img(
    #     "./ShadowMagic_output_shadows/image7_color.png", 
    #     "./ShadowMagic_output_shadows/image7_color_right_shadow3.png", 
    #     "./ShadowMagic_output_shadows/image7_line.png", 
    #     "./ShadowMagic_output_shadows/image7_flat.png")

    # display_img(
    #     "./ShadowMagic_output_shadows/image59_color.png", 
    #     "./ShadowMagic_output_shadows/image59_color_right_shadow1.png", 
    #     "./ShadowMagic_output_shadows/image59_line.png", 
    #     "./ShadowMagic_output_shadows/image59_flat.png")

    # display_img(
    #     "./ShadowMagic_output_shadows/image143_color.png", 
    #     "./ShadowMagic_output_shadows/image143_color_left_shadow0.png", 
    #     "./ShadowMagic_output_shadows/image143_line.png", 
    #     "./ShadowMagic_output_shadows/image143_flat.png")

    # display_img(
    #     "./ShadowMagic_output_shadows/image149_color.png", 
    #     "./ShadowMagic_output_shadows/image149_color_top_shadow2.png", 
    #     "./ShadowMagic_output_shadows/image149_line.png", 
    #     "./ShadowMagic_output_shadows/image149_flat.png")

    # display_img(
    #     "./ShadowMagic_output_shadows/image164_color.png", 
    #     "./ShadowMagic_output_shadows/image164_color_right_shadow0.png", 
    #     "./ShadowMagic_output_shadows/image164_line.png", 
    #     "./ShadowMagic_output_shadows/image164_flat.png")

    # display_img(
    #     "./ShadowMagic_output_shadows/image197_color.png", 
    #     "./ShadowMagic_output_shadows/image197_color_left_shadow2.png", 
    #     "./ShadowMagic_output_shadows/image197_line.png", 
    #     "./ShadowMagic_output_shadows/image197_flat.png")

    # display_img(
    #     "./ShadowMagic_output_shadows/image228_color.png", 
    #     "./ShadowMagic_output_shadows/image228_color_right_shadow1.png", 
    #     "./ShadowMagic_output_shadows/image228_line.png", 
    #     "./ShadowMagic_output_shadows/image228_flat.png")
    
    # display_img(
    #     "./ShadowMagic_output_shadows/image258_color.png", 
    #     "./ShadowMagic_output_shadows/image258_color_left_shadow2.png", 
    #     "./ShadowMagic_output_shadows/image258_line.png", 
    #     "./ShadowMagic_output_shadows/image258_flat.png")
    # display_img(
    #     "./ShadowMagic_output_shadows/image279_color.png", 
    #     "./ShadowMagic_output_shadows/image279_color_left_shadow3.png", 
    #     "./ShadowMagic_output_shadows/image279_line.png", 
    #     "./ShadowMagic_output_shadows/image279_flat.png")