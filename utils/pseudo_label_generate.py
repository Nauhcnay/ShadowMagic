import json
import os
import numpy as np
import argparse
import cv2

from PIL import Image
from os.path import join, exists
from misc import remove_alpha
from thinning import thinning
from preprocess import flat_to_fillmap, fillmap_to_color, get_bbox

# https://copyprogramming.com/howto/object-of-type-ndarray-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help = 'the input folder', default = '../experiments/03.Pesudo Labell')
    parser.add_argument('-o', help = 'the output folder', default = None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    # init result dictionary
    res = {
        'info': {'description': 'ShadowMagic pesudo label'}, 
        'images': [],
        'annotations': [],
        'categories':[{'id':1,'name':'Hair'}]}

    # scan and build up the labels
    idx = 0
    idx_c = 0
    for pimg in os.listdir(args.i):
        if 'flat' not in pimg: continue
        flat = np.array(Image.open(join(args.i, pimg)))
        fill, colors = flat_to_fillmap(flat)
        flat = remove_alpha(flat)      
        fill = thinning(fill)
        # write image info
        h, w = flat.shape[0], flat.shape[1]
        img_info = {'id': idx + 1, 'width': w, 'height': h, 'file_name': pimg}
        res['images'].append(img_info)
        idx += 1
        for region in np.unique(fill):
            if region == 0: continue
            contours, _ = cv2.findContours((fill == region).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for j in range(len(contours)):
                c = contours[j].squeeze()
                c_ = c.copy()
                if len(c.shape) == 1: continue
                c[:, [0, 1]] = c[:, [1, 0]]
                t, l, b, r = get_bbox(c, True)
                # The COCO bounding box format is [top left x position, top left y position, width, height]
                contour = {
                    'id':idx_c, 'iscrowd':0, 'image_id':idx, 'category_id': 1, 
                    'segmentation':[c_.flatten().tolist()], 'bbox':[l, t, r - l, b - t],
                    'area': (fill==region).sum(), 'categories':[{'id':1,'name':'Hair'}]}
                res['annotations'].append(contour)
                idx_c += 1
            # Image.fromarray(cv2.drawContours(flat, contours, -1, color = (0, 255, 0), thickness = 0 ).astype(np.uint8)).show()
            break
    if args.o is None:
        out_path = args.i
    else:
        out_path = args.o

    with open(join(out_path, "labels.json"), 'w') as f:
        f.write(json.dumps(res, cls=NumpyEncoder))
