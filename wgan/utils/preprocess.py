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
try:
    from utils.thinning import thinning
except:
    from thinning import thinning

from os.path import join, splitext, split, exists

# we need convert the key word in the file name to directions
KEY_TO_DIR = {"type1":"right", "type2":"left", "type3":"back", "type4":"top", "오":"right", "왼":"left",
    "역": "back", "위":"top", "l":"left", "r":"right", "긍": "right", "긍정":"right","부":"left", "부정":"left"}

# convert layer name to english
KOR_TO_ENG = {"선화":"line", "레이어":"shadow", "밑색":"flat", "용지":"background", "배경":"background",
    "레이어 1":"background", "레이어 2":"background", "layer 1":"background", "layer 2":"background", "그림자":"shadow", 
    "인물 밑색":"flat", "펜터치":"line"}
KOR_TO_ENG_S = {"선화":"line", "레이어":"shadow", "밑색":"flat", "용지":"background", "레이어 2":"shadow", "레이어 1":"background","그림자":"shadow"}

NAME_LIST = ["FULL01", "FULL02", "FULL03", "FULL04", "FULL05", "FULL06", "FULL07", "FULL08", "FACE03"]

def is_different_lname(fname):
    # there are some exceptions that we have to parse in a different rule
    for name in NAME_LIST:
        if name in fname: return True
    return False

def layer_to_png(psd, i, lname, png, size):
    h, w = size
    if "background" in lname or "bakcground" in lname: return None# we don't need background
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
            if "D3" in path or "D4" in path or "D5" in path:
                light_dir = KEY_TO_DIR[name.split("_")[2].lower()]
            else:
                light_dir = KEY_TO_DIR[name.split("_")[2].lower().strip(" ")]                
                # raise ValueError("the current folder has not been ready for parsing")
            psd = join(path, psd)   
            if debug: # if debug, keep the original file name
                png = join(path_pngs, name + ".png")
            else:    
                png = join(path_pngs, "%04d_%s.png"%(counter, light_dir))
            w, h = psd_f.width, psd_f.height
            # w, h = psd_f[0].size
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
                        if is_different_lname(name) or "D4" in path:
                            lname = KOR_TO_ENG_S.get(psd_f[i].name.lower(), psd_f[i].name.lower()).lower()
                            layer_to_png(psd_f, i, lname, png, (h, w))
                        else:
                            lname = KOR_TO_ENG.get(psd_f[i].name.lower(), psd_f[i].name.lower()).lower()
                            layer_to_png(psd_f, i, lname, png, (h, w))
            counter += 1

def flat_to_fillmap(flat, second_pass = True):
    '''
    flat png to fillmap, we need to consider flat always
        have alpha channel
    '''
    # print("Log:\tconverting flat PNG to fill map")
    h, w = flat.shape[0], flat.shape[1]
    r, g, b = flat[:,:,0], flat[:,:,1], flat[:,:,2]# split to r, g, b channel
    color_channel = r * 1e6 + g * 1e3 + b
    if flat.shape[-1] == 4:
        alpha_channel = flat[:,:,3]
    else:
        alpha_channel = np.logical_or(np.logical_or(r != 255, g != 255), b != 255)
    color_channel[alpha_channel == 0] = -1
    fill = np.ones((h,w)) # assume transparent region as 0
    fill[alpha_channel == 0] = 0 # fill region starts at 2
    color_idx = 2
    color_map = [np.array([0,0,0,0]), np.array([0,0,0,255])]
    th1 = h * w *5e-6
    th2 = h * w *5e-4
    # first pass
    for c in np.unique(color_channel):
        if c == -1: continue # skip transparent color
        if c == 0: continue # skip black color, it should be the line drawing
        # check if this region should be transparent
        mask = (color_channel == c).squeeze()
        # find bbox of the mask
        t, l, b, r = get_bbox(mask)
        mask_bsize = (r - l) * (b - t)
        if mask.sum() < th1 or (mask_bsize > 10 * mask.sum() and mask.sum() < th2):
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
    # second pass
    if second_pass:
        remove_stray_in_fill(fill)
    return fill.astype(int), np.array(color_map).astype(np.uint8)

def remove_stray_in_fill(fill):
    for c in np.unique(fill):
        if c == 1 or c == -1 or c == 0: continue
        masks = (fill == c).astype(np.uint8)
        _, regions = cv2.connectedComponents(masks, connectivity = 8)
        for r in np.unique(regions):
            if r == 0: continue
            rmask = regions == r
            if rmask.sum() <= 16:
                fill[rmask] = 1

def get_bbox(mask, pt_mode = False):
    # mask should be a boolean array
    if pt_mode:
        pts = mask
    else:
        pts = np.array(np.where(mask)).T
    left = pts[:, 1].min()
    right = pts[:, 1].max()
    top = pts[:, 0].min()
    bottom = pts[:, 0].max()
    return top, left, bottom, right

def shadow_to_fillmap(shadow):
    # print("Log:\tconverting shadow to fill map...")
    h, w = shadow.shape
    fill = np.ones((h,w))
    color_idx = 2
    color_map = [np.array(0), np.array(0)]
    for c in np.unique(shadow):
        mask = (shadow == c).squeeze()
        fill[mask] = color_idx
        color_map.append(np.array(c).astype(np.uint8))
        color_idx += 1
    return fill.astype(int), np.array(color_map).astype(np.uint8)

def fillmap_to_color(fill, color_map=None):
    if color_map is None:
        color_map = np.random.randint(0, 255, (fill.max() + 1, 3), dtype=np.uint8)
        r, c = np.unique(fill, return_counts = True)
        color_map[r[np.argsort(c)[-1]]] = [255, 255, 255]
    return color_map[fill], color_map

def int_to_color(color):
    r = int(color//1e6)
    g = int(color%1e6//1e3)
    b = int(color % 1000)
    return np.array([r, g, b, 255])

def flat_refine(flat, line = None, second_pass = True):
    ''' 
    Given: 
        flat, a numpy array as the flat png
        line, a numpy array as the line art
    Return:
        res, a numpy array that have the clear flat region boundary
    '''
    # convert line from rgba to bool mask
    # and interestingly, they use alpha channel as the grayscal image...
    
    if line is None:
        line_gray = np.ones(flat.shape) * 255
    else:
        if line.shape[-1] == 4:
            line_gray = 255 - line[:,:,3]
        if line.shape[-1] == 3:
            line_gray = line.mean(axis = -1)
        else:
            line_gray = line
    _, line_gray = cv2.threshold(line_gray, 127, 255, cv2.THRESH_BINARY)
    # line_gray = cv2.adaptiveThreshold(line_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # kernel = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype(np.uint8)
    # if second_pass == True:
    #     line_gray = cv2.erode(line_gray, kernel, iterations = 1) 
    if flat.shape[-1] == 4:
        flat_a = flat[:,:,3]
        flat[:,:,:3] = flat[:,:,:3] * np.repeat(np.expand_dims((flat_a == 255).astype(int), -1),3,-1)
    # convert flat to fill map
    fill, color_map = flat_to_fillmap(flat, second_pass)
    # set line drawing into the fill map
    fill[(255 - line_gray).astype(bool)] = 1
    ## remove stray regions
    for r in np.unique(fill):
        mask = fill == r
        if mask.sum() < 10:
            fill[mask] = 1
    fill = thinning(fill)
    flat_refined, _ =  fillmap_to_color(fill, color_map)
    return flat_refined, fill

def grayscale_shadow(shadow):
    bshadow = 255 - shadow[:, :, 3]
    return bshadow

def mask_shadow(fillmap, shadow, line, soft = False):
    line = line[:,:,3]
    line_thres = cv2.adaptiveThreshold(line, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,0)
    shadow = grayscale_shadow(shadow)
    mask = fillmap == 0
    shadow[mask.astype(bool)] = 255
    if soft == False:
        _, shadow = cv2.threshold(shadow,200,255,cv2.THRESH_BINARY)
        # remove stray shadows
        fill, color_map = shadow_to_fillmap(shadow)
        fill[line_thres==255] = 1
        remove_stray_in_fill(fill)
        fill = thinning(fill)
        shadow = fillmap_to_color(fill, color_map)
    return shadow

def png_refine(path_pngs, path_output):
    for png in os.listdir(path_pngs):
        if "0269" in png or "0270" in png or "0275" in png or "0276" in png: continue
        if "flat" not in png: continue
        print("log:\topening %s"%png)
        if os.path.exists(os.path.join(path_output, png)) and os.path.exists(os.path.join(path_output, png.replace("flat", "shadow"))) \
            and os.path.exists(os.path.join(path_output, png.replace("flat", "line"))):
            continue
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
        fill = None
        if os.path.exists(os.path.join(path_output, png)) is False:
            flat, fill = flat_refine(flat, edge)
            print("log:\trefined flat to %d regions"%len(np.unique(fill)))
            Image.fromarray(flat).save(os.path.join(path_output, png))
        if os.path.exists(os.path.join(path_output, png.replace("flat", "shadow"))) is False:
            if os.path.exists(os.path.join(path_output, png)) is True and fill is None:
                flat = np.array(Image.open(os.path.join(path_output, png)))
                _, fill = flat_refine(flat, edge, False)
                print("log:\topenning flat with %d regions"%len(np.unique(fill)))
            # print("log:\trefining shadow")
            shadow_full = mask_shadow(fill, shadow, line, False)
            Image.fromarray(shadow_full).save(os.path.join(path_output, png.replace("flat", "shadow")))
        # shadow_split = mask_shadow(fill, shadow, line, True)
        # save to results to target folder
        Image.fromarray(line).save(os.path.join(path_output, png.replace("flat", "line")))

if __name__ == "__main__":
    # '''psd layer to separate png images'''
    # PATH_TO_PSD = ["../dataset/raw/Natural", "../dataset/raw/NEW", "../dataset/raw/REFINED"]
    # PATH_TO_PSD = ["../dataset/raw/D8"]
    # OUT_PATH = "../dataset/Natural_png_rough"
    # psd_to_pngs(PATH_TO_PSD, OUT_PATH, 661, debug = False)
    

    '''correct shading layers'''
    # OUT_PATH = "../dataset/Natural_png_rough"
    # REFINED_PATH = "../dataset/Natural_png_rough_refined"
    # png_refine(OUT_PATH, REFINED_PATH)

    # debug for flat region refine
    flat = np.array(Image.open("image0111_flat.png"))
    line = np.array(Image.open("image0111_line.png"))
    flat_refined, _ = flat_refine(flat, line)
    Image.fromarray(flat_refined).save("res.png")
