'''
Misc functions for pre-processing or post-processing
'''

import json
import numpy as np
import cv2
from os.path import split, splitext
from PIL import Image
from preprocess import fillmap_to_color, flat_to_fillmap

# let's do this 
def to_hint_layer(flat, label):
    '''
    Given:
        flat, numpy array as the flat layer image
        label, dictionary as the semantic region labels
    Return:
        hint, numpy array as the hint line layer image
    '''
    limegreen = color_hex_to_dec("#32CD32")
    hint = np.zeros(flat.shape)
    fillmap, colors = flat_to_fillmap(flat)
    # merge colors by label
    for k, region in label.items():
        if k == "file": continue
        merge_colors(fillmap, colors, region)
    flat_new = fillmap_to_color(fillmap, colors)
    # draw lines 
    edges = cv2.Canny(flat_new, 100, 200)
    mask = edges == 255
    hint[mask] = limegreen
    # flat_new[mask] = limegreen
    return hint.astype(np.uint8)

def merge_colors(fillmap, colors, region):
    '''this is a inplace function!'''
    merge_color = None
    for sc in region:
        sc = color_hex_to_dec(sc)
        for i in range(len(colors)):
            dc = colors[i].tolist()
            if sc == dc:
                if merge_color is None:
                    merge_color = sc
                else:
                    # we merge the fill map by merging the color to the same
                    colors[i] = merge_color

def color_hex_to_dec(color):
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:], 16)
    return [r, g, b, 255]

if __name__ == "__main__":
    f_json = "../samples/jsonExample.json"
    f_flat = "../samples/flat png/0004_back_flat.png"
    # load img
    flat = np.array(Image.open(f_flat))
    _, name = split(f_flat)
    name, _ = splitext(name)
    # load json
    with open(f_json, 'r') as f:
        labels = json.load(f)
    # find label
    label = None
    for l in labels:
        if l["file"] == name:
            label = l
            break
    # show the hint layer
    Image.fromarray(to_hint_layer(flat, label)).show()