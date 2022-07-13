'''
1. psd layers to separate png images
2. correct shading layers
3. shading layers to real GT
4. create training dataset
'''
import os 
import threading
import subprocess
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from tqdm import tqdm

from os.path import join, splitext, split, exists

class RunCmd(threading.Thread):
    def __init__(self, cmd, timeout = 10):
        threading.Thread.__init__(self)
        self.cmd = cmd
        self.timeout = timeout

    def run(self):
        self.p = subprocess.Popen(self.cmd)
        self.p.wait()

    def Run(self):
        self.start()
        self.join(self.timeout)

        if self.is_alive():
            self.p.terminate()      #use self.p.kill() if process needs a kill -9
            self.join()

def psd_to_pngs(path_psd, path_pngs):
    if exists(path_pngs) is False: os.makedirs(path_pngs)
    cmd1 = "magick" 
    cmd2 = "convert"
    for psd in tqdm(os.listdir(path_psd)):
        psd_f = PSDImage.open(join(path_psd, psd))
        w, h = psd_f.size
        name, _ = splitext(psd)
        psd = join(path_psd, psd)   
        png = join(path_pngs, name+".png")
        # this will produce layer pngs without offset
        # RunCmd([cmd1, cmd2, psd, png]).Run()
        for i in range(len(psd_f)):
            img = psd_f[i].numpy()
            # get offset of current layer
            offset = psd_f[i].offset
            if len(offset) == 2:
                l, t = offset
                b = h
                r = w
            else:
                l, t, r, b = offset
            # pad each layer before saving
            res = np.ones((h, w, 4)) * 255
            res[:, :, 3] = 0 # set alpha channel to transparent by default
            if w < img.shape[1] or h < img.shape[0]: img = img[0:h, 0:w] # sometimes the layer could even larger than the artboard!
            res[t:b, l:r, :] = img * 255
            Image.fromarray(res.astype(np.uint8)).save(png.replace(".png", "_%d.png"%i))
if __name__ == "__main__":
    '''psd layer to separate png images'''
    PATH_TO_PSD = "../dataset/raw/Natural"
    OUT_PATH = "../dataset/train/raw_pngs"
    psd_to_pngs(PATH_TO_PSD, OUT_PATH)

    '''correct shading layers'''

