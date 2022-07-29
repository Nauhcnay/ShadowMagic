import numpy as np
import torch
import logging
import cv2

from os.path import exists, join, split, splitext
from os import listdir
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class BasicDataset(Dataset):
    
    def __init__(self, img_path, crop_size = 256):
        self.img_path = img_path
        self.crop_size = crop_size
        # we won't resize the image now, let's see how it will works
        self.resize = 1024
        # scan the file list if necessary
        if exists(join(img_path, "img_list.txt")) == False:
            self.scan_imgs()
        with open(join(img_path, "img_list.txt"), 'r') as f:
            self.ids = f.readlines()
        self.length = len(self.ids)
        # set dirction dict, mapping text to float number
        self.to_dir_label = {"right": 0.25, "left":0.5, "back":0.75, "top":1.0}
        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def scan_imgs(self):
        # helper function to scan the full dataset
        print("Log:\tscan the %s"%self.img_path)
        imgs = []
        for img in os.listdir(self.img_path):
            if "line" in img:
                if exists(join(self.img_path, img.replace("line", "flat"))) and\
                    exists(join(self.img_path, img.replace("line", "shadow"))):
                    imgs.append(join(self.img_path, img))
        with open("img_list.txt", 'w') as f:
            f.write('\n'.join(imgs))
        print("Log:\tdone")

    def remove_alpha(self, img, gray = False):
        # assum the img is numpy array
        # but we should be able to deal with different kind input images
        if len(img.shape) == 3:
            h, w, c = img.shape
            if c == 4:
                alpha = img[:, :, 3]
                whit_bg = np.ones((h, w, 3)) * 255
                img_res = img * alpha + whit_bg * (1 - alpha)
                if gray:
                    img_res = img_res.mean(axis = -1)
            else:
                img_res = img
        else:
            img_res = img
        return img_res

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        '''
            alright, here we need to several things...
            1. resize each input image
            2. remove alpha channel if it exsits
            3. merge line and flat layer, we will send them to the network together
            4. what kind of augmentation could we use? cause I guess flipping will not work...
            wait, or will? how about we also flip the label!
        '''
        # get image path
        line_path = self.ids[i]
        flat_path = line_path.replace("line", "flat")
        shad_path = line_path.replace("line", "shadow")

        # get light label
        _, n = split(line_path)
        label = n.split("_")[1]
        label = self.to_dir_label[label]

        # open images
        assert exists(line_path), \
            f'No line art found for the ID {idx}: {line_path}'
        assert exists(flat_path), \
            f'No flat found for the ID {idx}: {flat_path}'
        assert exists(shad_path), \
            f'No shadow found for the ID {idx}: {shad_path}'
        line_np = np.array(Image.open(line_path))
        flat_np = np.array(Image.open(flat_path))
        shad_np = np.array(Image.open(shap_path))
        
        # merge line and flat
        flat_np = self.remove_alpha(flat_np)
        line_th = cv2.adaptiveThreshold(line_np[:, :, 3] ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        line_mask = np.expand_dims(line_th == 0, axis = -1)
        flat_np[line_mask] = 0
        shad_np = self.remove_alpha(shad_np, gray = True)

        # augment image, let's do this in numpy!

        # convert to tensor, and the following process should all be done by cuda
        flat = self.to_tensor(flat_np)
        shad = self.to_tensor(shad_np / 255, False) # this is label infact
        
        # it returns tensor at last
        return flat, shad

    def to_tensor(self, pil_img, normalize = True):
        # assume the input is always grayscal
        if normalize:
            transforms = T.Compose(
                    [
                        # to tensor will change the channel order and divide 255 if necessary
                        T.ToTensor(),
                        T.Normalize(0.5, 0.5, inplace = True)
                    ]
                )
        else:
            transforms = T.Compose(
                    [
                        # to tensor will change the channel order and divide 255 if necessary
                        T.ToTensor(),
                    ]
                )

        return transforms(pil_img)