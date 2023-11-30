import os
import numpy as np
import cv2
from PIL import Image
from os.path import join, exists
from tqdm import tqdm

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

def crop_pad_resize(img, bbox, size = 512):
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



def gen_valset_sd():
    path_to_img = 'validation'


if __name__ == "__main__":
    # gen_trainset_sd()
    gen_valset_sd()
    pass