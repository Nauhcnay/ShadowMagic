import os
import numpy as np
import cv2
from PIL import Image
from os.path import join, exists
from tqdm import tqdm
import sys
from pathlib import Path as P
directory = os.path.realpath(os.path.dirname(__file__))
directory = str(P(directory).parent)
sys.path.append(join(directory, 'wgan', 'utils'))
from preprocess import flat_refine
import multiprocessing as mp
import random


def gen_prompt(dir, img_num, img_cond_num, color_mode):
    # one label example 
    # {"text": "pale golden rod circle with old lace background", "image": "images/0.png", "conditioning_image": "conditioning_images/0.png"}
    if color_mode == 'unchanged':
        text = "\"add shadow from %s lighting\""%dir
    elif color_mode == "add color":
        text = "\"add shadow from %s lighting and colorize\""%dir
    elif color_mode == "remove color":
        text = "\"add shadow from %s lighting and remove color\""%dir
    else:
        raise ValueError("invalid prompt mode%s"%color_mode)
    return "{\"text\": %s, \"image\": \"images/%05d.png\", \"conditioning_image\": \"conditioning_images/%05d.png\"}"%(text, img_num, img_cond_num)

def reverse_dir(light_dir):
    if light_dir == 'right':
        return 'left'
    elif light_dir == 'left':
        return 'right'

def write_to_file(out_cond, out_img, img_cond_num, img_num, color, color_shadowed, line, line_shadowed, light_dir, texts):
    # color to line
    assert exists(join(out_cond, "%05d.png"%img_cond_num)) is False
    assert exists(join(out_img, "%05d.png"%img_num)) is False
    Image.fromarray((color*255).astype(np.uint8)).save(join(out_cond, "%05d.png"%img_cond_num))
    Image.fromarray((line_shadowed*255).astype(np.uint8)).save(join(out_img, "%05d.png"%img_num))
    texts.append(gen_prompt(light_dir, img_num, img_cond_num, "remove color"))
    img_num += 1
    img_cond_num += 1

    # line to line
    assert exists(join(out_cond, "%05d.png"%img_cond_num)) is False
    assert exists(join(out_img, "%05d.png"%img_num)) is False
    Image.fromarray((line*255).astype(np.uint8)).save(join(out_cond, "%05d.png"%img_cond_num))
    Image.fromarray((line_shadowed*255).astype(np.uint8)).save(join(out_img, "%05d.png"%img_num))
    texts.append(gen_prompt(light_dir, img_num, img_cond_num, "unchanged"))
    img_num += 1
    img_cond_num += 1

    return img_num, img_cond_num

def get_bbox(flat):
    flat_ = flat.mean(axis = -1)
    coords = np.array(np.where(flat_ != 255)).T
    left = coords[:, 1].min()
    top = coords[:, 0].min()
    right = coords[:, 1].max()
    bottom = coords[:, 0].max()
    return (left, top, right, bottom)

def crop_pad_resize(img, bbox, size = 512, flat = False):
    # remove all back ground
    left, top, right, bottom = bbox    
    img = img[top:bottom, left:right, ...]
    h, w = img.shape[0], img.shape[1]
    new_size = h if h >= w else w
    img_ = (np.ones((new_size, new_size, 3)) * 255).astype(np.uint8)
    if h > w:
        pad = int((h - w) / 2)
        img_[:, pad:pad+w, ...] = img[...]
    elif w > h:
        pad = int((w - h) / 2)
        img_[pad:pad+h, :, ...] = img
    if flat:
        img_ = cv2.resize(img_, (size, size), interpolation = cv2.INTER_NEAREST)
    else:
        img_ = cv2.resize(img_, (size, size), interpolation = cv2.INTER_AREA)
    return img_

def gen_trainset_sd():
    input_path = "raw"
    output_path = "shadowmagic_xl"
    out_img_raw = join(output_path, 'images_raw')
    out_cond_raw = join(output_path, 'conditioning_images_raw')
    text_path = join(output_path, "train.jsonl")

    out_img = join(output_path, 'images')
    out_cond = join(output_path, 'conditioning_images')
    if exists(out_img_raw) == False: os.makedirs(out_img_raw)
    if exists(out_cond_raw) == False: os.makedirs(out_cond_raw)
    if exists(out_img) == False: os.makedirs(out_img)
    if exists(out_cond) == False: os.makedirs(out_cond)

    img_num = 0
    img_cond_num = 0
    texts = []
    for img in tqdm(os.listdir(input_path)):
        ## read input
        if "flat" not in img: continue
        assert exists(join(input_path, img.replace("flat", "line")))
        assert exists(join(input_path, img.replace("flat", "shadow")))
        # we only create color to shadow, line to shadow
        if exists(join(out_cond_raw, "%05d.png"%img_cond_num)):
            light_dir = img.split('_')[1]
            texts.append(gen_prompt(light_dir, img_num, img_cond_num, "remove color"))
            img_num += 1
            img_cond_num += 1
            texts.append(gen_prompt(light_dir, img_num, img_cond_num, "unchanged"))
            img_num += 1
            img_cond_num += 1
            if light_dir == 'left' or light_dir == 'right':
                light_dir = reverse_dir(light_dir)
                texts.append(gen_prompt(light_dir, img_num, img_cond_num, "remove color"))
                img_num += 1
                img_cond_num += 1
                texts.append(gen_prompt(light_dir, img_num, img_cond_num, "unchanged"))
                img_num += 1
                img_cond_num += 1
            continue
        flat = np.array(Image.open(join(input_path, img))).astype(float) / 255
        # add white back ground to flat
        flat_alpha = np.repeat(flat[..., -1][..., np.newaxis], 3, axis = -1)
        flat_color = flat[..., :3]
        flat_bg = np.ones(flat_color.shape).astype(float)
        flat = flat_alpha * flat_color + (1 - flat_alpha) * flat_bg
        # extract line from alpha channel
        line = np.array(Image.open(join(input_path, img.replace("flat", "line")))).astype(float) / 255
        line = np.repeat((1 - line[..., -1])[..., np.newaxis], 3, axis = -1)
        shadow = np.array(Image.open(join(input_path, img.replace("flat", "shadow")))).astype(float) / 255
        shadow = (shadow + 1) / 2 
        shadow = np.repeat(shadow[..., np.newaxis], 3, axis = -1)
        # create every variant from the reading
        color = flat * line
        color_shadowed = color * shadow
        line_shadowed = line * shadow

        ## read lighting direction
        light_dir = img.split('_')[1]
        ## convert to format of controlnet dataset
        img_num, img_cond_num = write_to_file(out_cond_raw, out_img_raw, img_cond_num, img_num, color, color_shadowed, line, line_shadowed, light_dir, texts)
        if light_dir == 'right' or light_dir == 'left':
            light_dir = reverse_dir(light_dir)
            color = np.fliplr(color)
            color_shadowed = np.fliplr(color_shadowed)
            line = np.fliplr(line)
            line_shadowed = np.fliplr(line_shadowed)
            img_num, img_cond_num = write_to_file(out_cond_raw, out_img_raw, img_cond_num, img_num, color, color_shadowed, line, line_shadowed, light_dir, texts)

    if exists(text_path) == False:
        with open(text_path, "w") as f:
            f.write('\n'.join(texts))

    ## then let's resize and pad
    img_list = os.listdir(out_cond_raw)
    img_list.sort()
    assert len(img_list) % 2 == 0

    for i in tqdm(range(0, len(img_list), 2)):
        
        lshadow1 = np.array(Image.open(join(out_img_raw, img_list[i])))
        bbox = get_bbox(lshadow1)
        lshadow1 = crop_pad_resize(lshadow1, bbox, size = 1024)
        Image.fromarray(lshadow1).save(join(out_img, img_list[i]))
        lshadow2 = np.array(Image.open(join(out_img_raw, img_list[i+1])))
        lshadow2 = crop_pad_resize(lshadow2, bbox, size = 1024)
        Image.fromarray(lshadow2).save(join(out_img, img_list[i+1]))

        color1 = np.array(Image.open(join(out_cond_raw, img_list[i])))
        color1 = crop_pad_resize(color1, bbox, size = 1024)
        Image.fromarray(color1).save(join(out_cond, img_list[i]))

        line1 = np.array(Image.open(join(out_cond_raw, img_list[i+1])))
        line1 = crop_pad_resize(line1, bbox, size = 1024)
        Image.fromarray(line1).save(join(out_cond, img_list[i+1]))

def remove_alpha(img):
    assert img.shape[-1] == 4
    alpha = (img[..., -1] / 255)[..., np.newaxis]
    bg = np.ones((img.shape[0], img.shape[1], 3)) * 255
    img = img[..., :3]
    return (img * alpha + bg * (1 - alpha)).astype(np.uint8)

def aq(name):
    return "\""+name+"\""

def gen_valset_sd_single(img, path_to_raw, path_to_sd15, path_to_sdxl):
    print("log:\topening %s"%img)
    val_root15 = './validation/sd1.5/'
    val_rootxl = './validation/sdxl/'
    directions = ['right', 'left', 'top', 'back']
    if exists(join(path_to_sdxl, img.replace('flat', 'color'))) == False:
        flat = np.array(Image.open(join(path_to_raw, img)))
        line = np.array(Image.open(join(path_to_raw, img.replace('flat', 'line'))))
        if line.shape[-1] == 4:
            line = 255 - line[:, :, 3] # use alpha channle only as our line 
        if flat.shape[-1] == 4:
            flat = remove_alpha(flat)
        # clean up flat image
        # print("log:\trefine the flat layer")
        flat_refined, _ = flat_refine(flat, line)
        flat_refined = remove_alpha(flat_refined)
        # blend flat and line 
        assert (flat.shape[0], flat.shape[1]) == (line.shape[0], line.shape[1])
        # print("log:\tblend flat and line layers")
        color = (((flat/255) * (line/255)) * 255).astype(np.uint8)
        # print("log:\tsave results to %s"%path_to_sd15)
        bbox = get_bbox(color)
        color_sd15 = crop_pad_resize(color, bbox, size = 512)
        line_sd15 = crop_pad_resize(line, bbox, size = 512)
        flat_sd15 = crop_pad_resize(flat_refined, bbox, size = 512, flat = True)
        Image.fromarray(color_sd15).save(join(path_to_sd15, img.replace('flat', 'color')))
        Image.fromarray(line_sd15).save(join(path_to_sd15, img.replace('flat', 'line')))
        Image.fromarray(flat_sd15).save(join(path_to_sd15, img))
        
        # print("log:\tsave results to %s"%path_to_sdxl)
        color_sd15 = crop_pad_resize(color, bbox, size = 1024)
        line_sd15 = crop_pad_resize(line, bbox, size = 1024)
        flat_sd15 = crop_pad_resize(flat_refined, bbox, size = 1024, flat = True)
        Image.fromarray(color_sd15).save(join(path_to_sdxl, img.replace('flat', 'color')))
        Image.fromarray(line_sd15).save(join(path_to_sdxl, img.replace('flat', 'line')))
        Image.fromarray(flat_sd15).save(join(path_to_sdxl, img))
    # generate validation cmds
    val_img = []
    direction = random.choices(directions, weights = (45, 45, 5, 5))[0]
    prompt_line = "add shadow from %s lighting"%direction
    direction = random.choices(directions, weights = (45, 45, 5, 5))[0]
    prompt_color = "add shadow from %s lighting and remove color"%direction
    val_img.append(['sd1.5', aq(val_root15+img.replace('flat', 'color')), aq(prompt_color)])
    val_img.append(['sd1.5', aq(val_root15+img.replace('flat', 'line')), aq(prompt_line)])
    val_img.append(['sdxl', aq(val_rootxl+img.replace('flat', 'color')), aq(prompt_color)])
    val_img.append(['sdxl', aq(val_rootxl+img.replace('flat', 'line')), aq(prompt_line)])
    return val_img

def gen_valset_sd():
    path_to_img = 'validation'
    ## generate the valdation set for both sd1.5 and sdxl
    path_to_raw = join(path_to_img, 'raw')
    path_to_sd15 = join(path_to_img, 'sd1.5')
    path_to_sdxl = join(path_to_img, 'sdxl')
    cmds = []
    for img in os.listdir(path_to_raw):
        if 'flat' not in img: continue
        cmds.append([img, path_to_raw, path_to_sd15, path_to_sdxl])
        # aa = gen_valset_sd_single(img, path_to_raw, path_to_sd15, path_to_sdxl)
        # import pdb
        # pdb.set_trace()
    with mp.Pool(8) as pool:
        res = pool.starmap(gen_valset_sd_single, cmds)

    val_img_15 = []
    val_prompt_15 = []
    val_img_xl = []
    val_prompt_xl = []
    for g in res:
        for l in g:
            if l[0] == 'sd1.5':
                val_img_15.append(l[1])
                val_prompt_15.append(l[2])
            elif l[0] == 'sdxl':
                val_img_xl.append(l[1])
                val_prompt_xl.append(l[2])
    with open(join(path_to_sd15, 'val.txt'), "w") as f:
        l1 = " ".join(val_img_15)
        l2 = " ".join(val_prompt_15)
        f.write("\n".join([l1, l2]))
    with open(join(path_to_sdxl, 'val.txt'), "w") as f:
        l1 = " ".join(val_img_xl)
        l2 = " ".join(val_prompt_xl)
        f.write("\n".join([l1, l2]))

if __name__ == "__main__":
    __spec__ = None
    # gen_trainset_sd()
    gen_valset_sd()