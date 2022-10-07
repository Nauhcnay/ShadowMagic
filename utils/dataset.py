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
    
    def __init__(self, img_path, crop_size = 256, resize = 1024, val = False, l1_loss = False):
        # we may need some validation result in the future
        self.val = val
        self.img_path = img_path
        self.crop_size = crop_size
        self.l1_loss = l1_loss
        # we won't resize the image now, let's see how it will works
        self.resize = resize
        # scan the file list if necessary
        if exists(join(img_path, "img_list.txt")) == False:
            self.scan_imgs()
        with open(join(img_path, "img_list.txt"), 'r') as f:
            self.ids = f.readlines()
        # split validation set, let's use 5% samples for validation
        val_idx = int(len(self.ids) * 0.95)
        if val:
            self.ids = self.ids[val_idx:]
        else:
            self.ids = self.ids[:val_idx]
        self.length = len(self.ids)
        # set dirction dict, mapping text to float number
        self.to_dir_label = {"right": 0.25, "left":0.5, "back":0.75, "top":1.0}
        self.lable_flip = {0.25:0.5, 0.5:0.25}
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def scan_imgs(self):
        # helper function to scan the full dataset
        print("Log:\tscan the %s"%self.img_path)
        imgs = []
        img_path = join(self.img_path, "img")
        for img in listdir(img_path):
            if "line" in img:
                if exists(join(img_path, img.replace("line", "flat"))) and\
                    exists(join(img_path, img.replace("line", "shadow"))):
                    imgs.append(join(img_path, img))
        with open(join(self.img_path, "img_list.txt"), 'w') as f:
            f.write('\n'.join(imgs))
        print("Log:\tdone")

    def remove_alpha(self, img, gray = False):
        # assum the img is numpy array
        # but we should be able to deal with different kind input images
        if len(img.shape) == 3:
            h, w, c = img.shape
            if c == 4:
                alpha = np.expand_dims(img[:, :, 3], -1) / 255
                whit_bg = np.ones((h, w, 3)) * 255
                img_res = img[:, :, :3] * alpha + whit_bg * (1 - alpha)
                if gray:
                    img_res = img_res.mean(axis = -1)
            else:
                img_res = img
        else:
            img_res = img
        return img_res

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
            alright, here we need to several things...
            1. resize each input image
            2. remove alpha channel if it exsits
            3. merge line and flat layer, we will send them to the network together
            4. what kind of augmentation could we use? cause I guess flipping will not work...
            wait, or will? how about we also flip the label!
        '''
        # get image path
        line_path = self.ids[idx].strip("\n")
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
        shad_np = np.array(Image.open(shad_path))
        
        # create training mask
        mask_np = flat_np[:, :, 3]
        _, mask_np = cv2.threshold(mask_np.squeeze(), 127, 255, cv2.THRESH_BINARY)

        # merge line and flat
        flat_np = self.remove_alpha(flat_np)
        flat_np = flat_np * (1 - np.expand_dims(line_np[:, :, 3], axis = -1) / 255)
        line_np = 255 - line_np[:, :, 3] # remove alpha channel, but yes, we use alpha channel as the line drawing
        shad_np = self.remove_alpha(shad_np, gray = True)
        # we need an additional thershold for this
        _, shad_np = cv2.threshold(shad_np, 127, 255, cv2.THRESH_BINARY)

        # create edge mask
        mask_edge_np = cv2.Canny(shad_np, 100, 200)
        mask_edge_np = cv2.dilate(mask_edge_np, self.kernel, iterations = 2)
        # resize image, now we still have to down sample the input a little bit for a easy training
        h, w = shad_np.shape
        h, w = self.resize_hw(h, w)
        flat_np = cv2.resize(flat_np, (w, h), interpolation = cv2.INTER_AREA)
        shad_np = cv2.resize(shad_np, (w, h), interpolation = cv2.INTER_NEAREST)
        mask_np = cv2.resize(mask_np, (w, h), interpolation = cv2.INTER_NEAREST)
        mask_edge_np = cv2.resize(mask_edge_np, (w, h), interpolation = cv2.INTER_NEAREST)

        # we don't need image augmentation for val
        # if True:
        if self.val == False:
            # augment image, let's do this in numpy!
            img_list, label = self.random_flip([flat_np, line_np, shad_np, mask_np, mask_edge_np], label)
            flat_np, line_np, shad_np, mask_np, mask_edge_np = img_list
            bbox = self.random_bbox(flat_np)
            flat_np, line_np, shad_np, mask_np, mask_edge_np = self.crop([flat_np, line_np, shad_np, mask_np, mask_edge_np], bbox)

        
        # clip values
        flat_np = flat_np.clip(0, 255)
        shad_np = shad_np.clip(0, 255)
        shad_np_d2x = self.down_sample(shad_np)
        shad_np_d4x = self.down_sample(shad_np_d2x)
        shad_np_d8x = self.down_sample(shad_np_d4x)
        mask_np = mask_np.clip(0, 255)
        mask_edge_np = mask_edge_np.clip(0, 255)

        # convert to tensor, and the following process should all be done by cuda
        flat = self.to_tensor(flat_np / 255)
        line = self.to_tensor(line_np.copy(), False)
        if self.l1_loss:
            shad = self.to_tensor(1 - shad_np / 255) # if we use l1 loss, let's treat the shading as image
        else:
            shad = self.to_tensor(1 - shad_np / 255, False)
            shad_d2x = self.to_tensor(1 - shad_np_d2x / 255, False)
            shad_d4x = self.to_tensor(1 - shad_np_d4x / 255, False)
            shad_d8x = self.to_tensor(1 - shad_np_d8x / 255, False)
        mask = self.to_tensor(mask_np / 255, False)
        mask_edge = self.to_tensor(mask_edge_np / 255, False)
        label = torch.Tensor([label])
        assert line.shape == shad.shape
        # it returns tensor at last
        return flat, line, (shad, shad_d2x, shad_d4x, shad_d8x), mask, mask_edge, label
    
    def down_sample(self, img):
        dw = int(img.shape[1] / 2)
        dh = int(img.shape[0] / 2)
        img_d2x = cv2.resize(img, (dw, dh), interpolation = cv2.INTER_NEAREST)
        return img_d2x
    
    def random_bbox(self, img):
        h, w, _ = img.shape
        # we generate top, left, bottom, right
        t = np.random.randint(0, h - self.crop_size - 1)
        l = np.random.randint(0, w - self.crop_size - 1)
        return (t, l, t + self.crop_size, l + self.crop_size)

    def crop(self, imgs, bbox):
        t, l, b, r = bbox
        res = []
        for img in imgs:
            res.append(img[t:b, l:r])
        return res

    def random_flip(self, imgs, label, p = 0.5):
        # we only consider the horizontal flip
        dice = np.random.uniform()
        if dice < p:
            flipped = []
            for img in imgs:
                # flip the image and label
                flipped.append(np.flip(img, axis = 1))
            label = self.lable_flip.get(label, label)
        else:
            # don't change anything
            flipped = imgs
        return flipped, label
    
    def resize_hw(self, h, w):
        # we resize the shorter edge to the target size
        if h > w:
            ratio =  h / w
            h = int(self.resize * ratio)
            w = self.resize
        else:
            ratio = w / h
            w = int(self.resize * ratio)
            h = self.resize
        return h, w

    def to_tensor(self, img_np, normalize = True):
        # assume the input is always grayscal
        if normalize:
            transforms = T.Compose(
                    [
                        T.ToTensor(),
                        T.Normalize(0.5, 0.5, inplace = True)
                    ]
                )
        else:
            transforms = T.Compose(
                    [
                        T.ToTensor()
                    ]
                )
        return transforms(img_np)
        