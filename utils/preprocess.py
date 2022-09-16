'''
1. psd layers to separate png images
2. correct shading layers
3. shading layers to real GT
4. create training dataset
'''
import os
import numpy as np
import cv2
from PIL import Image
from psd_tools import PSDImage
from tqdm import tqdm
from thinning import thinning

from os.path import join, splitext, split, exists

# we need convert the key word in the file name to directions
KEY_TO_DIR = {"type1":"right", "type2":"left", "type3":"back", "type4":"top", "오":"right", "왼":"left",
    "역": "back", "위":"top", "l":"left", "r":"right", "긍": "right", "긍정":"right","부":"left", "부정":"left"}

# convert layer name to english
KOR_TO_ENG = {"선화":"line", "레이어":"shadow", "밑색":"flat", "용지":"background", "배경":"background",
    "레이어 1":"shadow", "레이어 2":"background", "layer 1":"background", "layer 2":"background", "그림자":"shadow", "인물 밑색":"flat", "펜터치":"line"}
KOR_TO_ENG_S = {"선화":"line", "레이어":"shadow", "밑색":"flat", "용지":"background", "레이어 2":"shadow", "레이어 1":"background"}

NAME_LIST = ["FULL01", "FULL02", "FULL03", "FULL04", "FULL05", "FULL06", "FULL07", "FULL08", "FACE03"]

def is_different_lname(fname):
    # there are some exceptions that we have to parse in a different rule
    for name in NAME_LIST:
        if name in fname: return True
    return False

def layer_to_png(psd, i, lname, png, size):
    h, w = size
    if lname == "background": return None# we don't need background
    if os.path.exists(png.replace(".png", "_%s.png"%lname)): return None
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
        img = img[0:h-t, 0:w-l, :] # sometimes the layer could even larger than the artboard! then we need to crop from the bottom right
    if t < 0: # t or l could also be negative number, so in that case we need to crop from the top left, too
        img = img[-t:, :, :]
    if l < 0:  
        img = img[:, -l:, :]
    top = t if t > 0 else 0
    left = l if l > 0 else 0
    res[top:img.shape[0]+top, left:img.shape[1]+left, :] = img * 255
    Image.fromarray(res.astype(np.uint8)).save(png.replace(".png", "_%s.png"%lname))
    return None

def psd_to_pngs(path_psd, path_pngs, counter, debug = False):
    '''
    convert all psd file into raw png files, and rename each png file with numbers
    '''
    if exists(path_pngs) is False: os.makedirs(path_pngs)
    for path in path_psd:
        for psd in tqdm(os.listdir(path)):
            # read psd file
            psd_f = PSDImage.open(join(path, psd))
            name, _ = splitext(psd)
            
            # '''for debug'''
            # if "CLOTH10" in name:
            #     import pdb
            #     pdb.set_trace()
            # else:
            #     continue
            # '''for debug'''

            # there are multiple naming format so we need to deal with them separately
            # generally, we need to get light direction from the file name
            if "Natural" in path or "NEW" in path:
                light_dir = KEY_TO_DIR[name.split("_")[1].lower()]
            elif "REFINED" in path:
                light_dir = KEY_TO_DIR[name.split("_")[2].lower()]
            elif "D1" in path or "D2" in path:
                if "-" in name:
                    light_dir = KEY_TO_DIR[name.split("-")[-1].lower()]
                else:
                    light_dir = KEY_TO_DIR[name.split("_")[-1].lower()]
            if "D3" in path:
                light_dir = KEY_TO_DIR[name.split("_")[2].lower()]
            else:
                raise ValueError("the current folder has not been ready for parsing")
            psd = join(path, psd)   
            if debug: # if debug, keep the original file name
                png = join(path_pngs, name + ".png")
            else:    
                png = join(path_pngs, "%04d_%s.png"%(counter, light_dir))
            w, h = psd_f[0].size
            # read each layer in the opened psd file
            # the parse logic here is terrible...
            for i in range(len(psd_f)):    
                if "왼" in name and "NEW" in path and debug is False:
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
                    if debug:
                        # if we get layer group
                        if psd_f[i].is_group():
                            for j in range(len(psd_f[i])):
                                lname = psd_f[i][j].name.lower()
                                layer_to_png(psd_f[i], j, lname, png, (h, w))
                        else:
                            lname = psd_f[i].name.lower()
                            layer_to_png(psd_f, i, lname, png, (h, w))        
                    else:
                        if is_different_lname(name):
                            lname = KOR_TO_ENG_S.get(psd_f[i].name.lower(), psd_f[i].name.lower()).lower()
                            layer_to_png(psd_f, i, lname, png, (h, w))
                        else:
                            lname = KOR_TO_ENG.get(psd_f[i].name.lower(), psd_f[i].name.lower()).lower()
                            layer_to_png(psd_f, i, lname, png, (h, w))
            counter += 1

def flat_to_fillmap(flat):
    '''
    flat png to fillmap, we need to consider flat always
        have alpha channel
    '''
    print("Log:\tconverting flat PNG to fill map...")
    h, w = flat.shape[0], flat.shape[1]
    alpha_channel = flat[:,:,3]
    r, g, b = flat[:,:,0], flat[:,:,1], flat[:,:,2]# split to r, g, b channel
    color_channel = r * 1e6 + g * 1e3 + b
    color_channel[alpha_channel == 0] = -1
    fill = np.ones((h,w)) # assume transparent region as 0
    fill[alpha_channel == 0] = 0 # fill region starts at 2
    color_idx = 2
    color_map = [np.array([0,0,0,0]), np.array([0,0,0,255])]
    thresh_color = h * w *0.0005
    for c in tqdm(np.unique(color_channel)):
        if c == -1: continue # skip transparent color
        if c == 0: continue # skip black color, it should be the line drawing
        # check if this region should be transparent
        mask = (color_channel == c).squeeze()
        if mask.sum() < thresh_color:
            fill[mask] = 1
            continue
        c_alpha = fill[mask]
        # find the majority color under current mask in alpha channel
        c_alpha_major, count = np.unique(c_alpha, return_counts = True)
        c_alpha = c_alpha_major[np.argmax(count)]
        # skip this color if the region is transparent
        if c_alpha == 0: continue
        # record into the fillmap
        fill[mask] = color_idx
        # update color index and dict
        color_idx += 1
        color_map.append(int_to_color(c))
    return fill.astype(int), np.array(color_map).astype(np.uint8)

def shadow_to_fillmap(shadow):
    print("Log:\tconverting shadow to fill map...")
    h, w = shadow.shape
    fill = np.ones((h,w))
    color_idx = 2
    color_map = [np.array(0), np.array(0)]
    for c in tqdm(np.unique(shadow)):
        mask = (shadow == c).squeeze()
        fill[mask] = color_idx
        color_map.append(np.array(c).astype(np.uint8))
        color_idx += 1
    return fill.astype(int), np.array(color_map).astype(np.uint8)

def fillmap_to_color(fill, color_map=None):
    if color_map is None:
        color_map = np.random.randint(0, 255, (np.max(fill) + 1, 3), dtype=np.uint8)
        color_map[0] = [255, 255, 255]
        return color_map[fill]
    else:
        return color_map[fill]

def int_to_color(color):
    r = int(color//1e6)
    g = int(color%1e6//1e3)
    b = int(color % 1000)
    return np.array([r, g, b, 255])

def flat_refine(flat, line, second_pass = False):
    ''' 
    Given: 
        flat, a numpy array as the flat png
        line, a numpy array as the line art
    Return:
        res, a numpy array that have the clear flat region boundary
    '''
    # convert line from rgba to bool mask
    # and interestingly, they use alpha channel as the grayscal image...
    if len(line.shape) == 3:
        line_gray = 255 - line[:,:,3]
    else:
        line_gray = line
    _, line_gray = cv2.threshold(line_gray, 127, 255, cv2.THRESH_BINARY)
    # line_gray = cv2.adaptiveThreshold(line_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # kernel = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype(np.uint8)
    # if second_pass == True:
    #     line_gray = cv2.erode(line_gray, kernel, iterations = 1) 
    flat_a = flat[:,:,3]
    flat[:,:,:3] = flat[:,:,:3] * np.repeat(np.expand_dims((flat_a == 255).astype(int), -1),3,-1)
    # convert flat to fill map
    fill, color_map = flat_to_fillmap(flat)
    # set line drawing into the fill map
    fill[(255 - line_gray).astype(bool)] = 1
    # remove stray regions
    flat_refined =  fillmap_to_color(fill, color_map)
    for r in np.unique(fill):
        mask = fill == r
        if mask.sum() < 10:
            fill[mask] = 1
    fill = thinning(fill)
    flat_refined =  fillmap_to_color(fill, color_map)
    return flat_refined, fill

def binary_shadow(shadow):
    bshadow = 255 - shadow[:, :, 3]
    return bshadow

def mask_shadow(fillmap, shadow, line, split = False):
    line = line[:,:,3]
    line_thres = cv2.adaptiveThreshold(line, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,0)
    shadow = binary_shadow(shadow)

    if split is False:
        mask = fillmap == 0
        # kernel = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype(np.uint8)
        # mask = cv2.dilate(mask.astype(np.uint8)*255, kernel, iterations = 2)
        shadow[mask.astype(bool)] = 255
        _, shadow = cv2.threshold(shadow,200,255,cv2.THRESH_BINARY)
        # remove stray shadows
        fill, color_map = shadow_to_fillmap(shadow)
        fill[line_thres==255] = 1
        fill = thinning(fill)
        shadow = fillmap_to_color(fill, color_map)
        return shadow
    else:
        # not sure if this branch is necessary
        # split shadow region by flat region
        pass

def png_refine(path_pngs, path_output):
    for png in os.listdir(path_pngs):
        if "flat" not in png: continue
        flat = np.array(Image.open(os.path.join(path_pngs, png)))
        shadow = np.array(Image.open(os.path.join(path_pngs, png.replace("flat", "shadow"))))
        # open the flat image again to generate the line for flat region refinement
        edge = np.array(Image.open(os.path.join(path_pngs, png)))
        line = np.array(Image.open(os.path.join(path_pngs, png.replace("flat", "line"))))
        # we need manually convert RBGA image to the grayscale image
        edge_a = np.repeat(np.expand_dims(edge[:,:,-1], axis = -1), 3, axis = -1)
        edge_rgb = edge[:,:,:3]
        bg_white = np.ones(edge[:,:,:3].shape) * 255
        edge_rgb = (edge_rgb * (edge_a >= 250) + bg_white * (edge_a <250)).astype(np.uint8)
        edge_gray = cv2.cvtColor(edge_rgb, cv2.COLOR_BGR2GRAY)
        edge = np.bitwise_not(cv2.Canny(edge_gray,0,20).astype(bool)).astype(np.uint8)*255
        # if "0004_back_flat" in png:
        #     import pdb
        #     pdb.set_trace()
        # else:
        #     continue
        '''refine by the line drawings, but this may cause hairy edges '''
        # # open files
        # flat = np.array(Image.open(os.path.join(path_pngs, png)))
        # line = np.array(Image.open(os.path.join(path_pngs, png.replace("flat", "line"))))
        # shadow = np.array(Image.open(os.path.join(path_pngs, png.replace("flat", "shadow"))))
        # refine pngs  
        flat, fill = flat_refine(flat, edge)
        shadow_full = mask_shadow(fill, shadow, line)
        # shadow_split = mask_shadow(fill, shadow, line, True)
        # save to results to target folder
        Image.fromarray(line).save(os.path.join(path_output, png.replace("flat", "line")))
        Image.fromarray(flat).save(os.path.join(path_output, png))
        Image.fromarray(shadow_full).save(os.path.join(path_output, png.replace("flat", "shadow")))

if __name__ == "__main__":
    # '''psd layer to separate png images'''
    # PATH_TO_PSD = ["../dataset/raw/Natural", "../dataset/raw/NEW", "../dataset/raw/REFINED"]
    # PATH_TO_PSD = ["../dataset/raw/D3"]
    # OUT_PATH = "../dataset/Natural_png_rough"
    # psd_to_pngs(PATH_TO_PSD, OUT_PATH, 255, debug = False)
    

    '''correct shading layers'''
    OUT_PATH = "../dataset/Natural_png_rough"
    REFINED_PATH = "../dataset/Natural_png_rough_refined"
    png_refine(OUT_PATH, REFINED_PATH)


