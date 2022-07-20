'''
1. psd layers to separate png images
2. correct shading layers
3. shading layers to real GT
4. create training dataset
'''
import os 
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from tqdm import tqdm

from os.path import join, splitext, split, exists

# we need convert the key word in the file name to directions
KEY_TO_DIR = {"type1":"right", "type2":"left", "type3":"back", "type4":"top", "오":"right", "왼":"left", 
    "역": "back", "위":"top"}

# convert layer name to english
KOR_TO_ENG = {"선화":"line", "레이어":"shadow", "밑색":"flat", "용지":"background", "레이어 2":"background"}

def layer_to_png(psd, i, lname, png, size):
    h, w = size
    if lname == "background": return # we don't need background
    img = psd[i].numpy()
    # get offset of current layer
    offset = psd[i].offset
    if len(offset) == 2:
        l, t = offset
        b = h
        r = w
    else:
        l, t, r, b = offset
    # pad each layer before saving
    res = np.ones((h, w, 4)) * 255
    res[:, :, 3] = 0 # set alpha channel to transparent by default
    if (w - l) < img.shape[1] or (h - t) < img.shape[0]: 
        img = img[0:h-t, 0:w-l, :] # sometimes the layer could even larger than the artboard!
    if t < 0 or l < 0: # t or l could also be negative number...
        img = img[:-t if t < 0 else t, :-l if l < 0 else l, :]
    top = t if t > 0 else 0
    left = l if l > 0 else 0
    res[top:img.shape[0]+top, left:img.shape[1]+left, :] = img * 255
    Image.fromarray(res.astype(np.uint8)).save(png.replace(".png", "_%s.png"%lname))

def psd_to_pngs(path_psd, path_pngs, counter):
    '''
    convert all psd file into raw png files, and rename each png file with numbers
    '''
    if exists(path_pngs) is False: os.makedirs(path_pngs)
    for path in path_psd:
        for psd in tqdm(os.listdir(path)):
            # read psd file
            psd_f = PSDImage.open(join(path, psd))
            name, _ = splitext(psd)
            # there are multiple naming format so we need to deal with them separately
            # generally, we need to get light direction from the file name
            if "Natural" in path or "NEW" in path:
                light_dir = KEY_TO_DIR[name.split("_")[1].lower()]
            elif "REFINED" in path:
                light_dir = KEY_TO_DIR[name.split("_")[2].lower()]
            else:
                raise ValueError("the current folder has not been ready for parsing")
            psd = join(path, psd)   
            png = join(path_pngs, "%04d_%s.png"%(counter, light_dir))
            w, h = psd_f[0].size
            # read each layer in the opened psd file
            for i in range(len(psd_f)):
                if "왼" in name and "NEW" in path:
                    if i == len(psd_f) - 1: 
                        lname = "line"
                        layer_to_png(psd_f, i, lname, png, (h, w))
                    elif i == len(psd_f) - 2:
                        for j in range(len(psd_f[i])):
                            if j == len(psd_f[i]) - 1:
                                lname = "shadow"
                                layer_to_png(psd_f[i], j, lname, png, (h, w))
                            elif j == len(psd_f[i]) - 2:
                                lname = "flat"
                                layer_to_png(psd_f[i], j, lname, png, (h, w))
                else:       
                    lname = KOR_TO_ENG.get(psd_f[i].name.lower(), psd_f[i].name.lower())
                    layer_to_png(psd_f, i, lname, png, (h, w))
            counter += 1
if __name__ == "__main__":
    '''psd layer to separate png images'''
    # PATH_TO_PSD = ["../dataset/raw/Natural", "../dataset/raw/NEW", "../dataset/raw/REFINED"]
    PATH_TO_PSD = ["../dataset/raw/Natural", "../dataset/raw/REFINED"]
    OUT_PATH = "../dataset/raw_png"
    psd_to_pngs(PATH_TO_PSD, OUT_PATH, 0)

    '''correct shading layers'''

