# refine the shadow and make it controlable 
import os
import cv2
import numpy as np

from PIL import Image
from scipy.signal import convolve2d as conv

K = np.array([
        [0,1,0], 
        [1,1,1], 
        [0,1,0], 
        ]).astype(float)

def display_img(img, shadow, line, resize_to = 1024):
    # read images
    if isinstance(img, str):
        img = cv2.imread(img)
    if isinstance(shadow, str):
        shadow = cv2.imread(shadow)
    if isinstance(line, str):
        line = cv2.imread(line)
    if len(shadow.shape) > 2:
        shadow = shadow.mean(axis = -1)
    # turn line into a mask which indicates the shadow boundary
    if len(line.shape) > 2:
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
        shadow = cv2.resize(shadow.astype(np.uint8), (w_new, h_new), interpolation = cv2.INTER_AREA)
        line = cv2.resize(line.astype(np.uint8), (w_new, h_new), interpolation = cv2.INTER_AREA).astype(float)
    while True:
        res = (img * to_blend_shadow(shadow)[..., np.newaxis]).astype(np.uint8)
        cv2.imshow('results',res)
        k = cv2.waitKey(100)
        if k == ord('a'):
            shadow = decrease_shadow(shadow, line)
        elif k == ord('d'):
            shadow = increase_shadow(shadow, line)
        elif k == ord('q'):
            break
    cv2.destroyAllWindows()

def increase_shadow(shadow, line):
    # line = cv2.dilate((line*255).astype(np.uint8), K.astype(np.uint8))
    shadow = shadow.copy()
    added = np.zeros(shadow.shape).astype(bool)
    shadow_masked = shadow.astype(float) + line*255
    shadow_masked = conv(shadow_masked, K, mode='same')
    added[(shadow_masked >0) & (shadow_masked < 5)] = True
    shadow[added] = True
    return shadow

def decrease_shadow(shadow, line):
    # line = cv2.dilate((line*255).astype(np.uint8), K.astype(np.uint8))
    shadow = shadow.copy()
    removed = np.zeros(shadow.shape).astype(bool)
    shadow_masked = shadow.astype(float) + line*255
    shadow_masked = conv(shadow_masked, K, mode='same')
    removed[(shadow_masked > 0) & (shadow_masked < 5)] = True
    shadow[removed] = False
    return shadow

def to_blend_shadow(shadow, thr = 0.9):
    res = np.zeros(shadow.shape)
    res[shadow >= thr] = 0.5
    res[shadow < thr] = 1
    return res

def to_binary_shadow(shadow):
    return shadow != 255  

# open image
# show image
display_img("./results/image143_color.png", "./results/image143_color_left_shadow1.png", "./results/image143_line.png")
