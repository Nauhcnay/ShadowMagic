import argparse
import logging
import os
import sys
import cv2

import numpy as np
import torch
import torch.nn as nn

# let's add the wandb
import wandb
from datetime import datetime

from torch import optim
from tqdm import tqdm
from layers import UNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torch.nn import functional as F
from torchvision import utils
from PIL import Image

# let's add anisotropic penalty
def get_ij_kernel(i, j, size = 3):
    assert i < size ** 2
    assert j < size ** 2
    assert i != j
    k = np.zeros((3, 3))
    k[i//3][i%3] = 1
    k[j//3][j%3] = -1
    return k

def get_ap_kernel(size = 3):
    '''
    Given,
        size, a integer for the convlution window size
    Return 
        a numpy array that is used for the anisotropic penalty convlution kernel
    '''
    kernels = []
    for i in range(size**2):
        for j in range(i+1, size**2):
            kernels.append(get_ij_kernel(i, j, size = size))
    kernel = np.stack(kernels, axis = 0)
    return np.expand_dims(kernel, axis = 1)

def anisotropic_penalty(pre, line, size = 3, k = 1):
    '''
    compute the anisotropic penalty in paper:
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_SmartShadow_Artistic_Shadow_Drawing_Tool_for_Line_Drawings_ICCV_2021_paper.pdf
    '''
    pre = torch.sigmoid(pre)
    ap_kernel = get_ap_kernel(size)
    ap_kernel = torch.Tensor(ap_kernel).float().to(pre.device)
    pre_ap = F.conv2d(pre, ap_kernel, padding = 'same')
    pre_ap = pre_ap.pow(2).sum(dim = 1)
    line_ap = F.conv2d(line, ap_kernel, padding = 'same')
    line_ap = line_ap.pow(2).sum(dim = 1)
    line_ap = torch.exp(-line_ap/k**2)
    loss = (pre_ap * line_ap).sum()
    return loss
    
def weighted_bce_loss(pre, target):
    # compute loss map
    bce_loss = F.binary_cross_entropy_with_logits(pre, target, reduction = 'none')
    # compute loss mask
    weights = [1, 1, 0.1, 0.1]
    mask_pre = torch.sigmoid(pre) > 0.5
    mask_pos = target.bool()
    mask_neg = torch.logical_not(mask_pos)
    mask_fn = torch.logical_and(torch.logical_not(mask_pre), mask_pos)
    mask_fp = torch.logical_and(mask_pre, mask_neg)
    masks = [mask_neg, mask_pos, mask_fp, mask_fn]
    # compute final loss
    loss = 0
    avg = 0
    for i in range(len(masks)):
        if masks[i].sum() > 0:
            loss = loss + bce_loss[masks[i]].mean() * weights[i] 
            avg += 1
    avg = 1 if avg == 0 else avg
    return loss / avg

def denormalize(img):
    # denormalize
    return (img / 2 + 0.5).clamp(0, 1)

def train_net(
              img_path,
              net,
              device,
              epochs=999,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              crop_size = 256,
              resize = 1024,
              l1_loss = False,
              name = None,
              drop_out = False,
              ap = False):

    # create dataloader
    dataset_train = BasicDataset(img_path, crop_size = crop_size, resize = resize, l1_loss = l1_loss)
    dataset_val = BasicDataset(img_path, crop_size = crop_size, resize = resize, val = True, l1_loss = l1_loss)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    # we don't need valiation currently
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # get dataset lenght
    n_train = len(dataset_train)
    
    # display train summarys
    global_step = 0
    

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Crop size:       {crop_size}
        Drop Out:        {drop_out}
        Loss:            {"L1" if l1_loss else "BCE"}
    ''')

    # not sure which optimizer will be better
    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [200, 400], gamma = 0.1, verbose = True)
    # create the loss function
    # the task is in fact a binary classification problem
    if l1_loss:
        criterion = nn.L1Loss()
    else:
        # let's use the focal loss instead of the BCE loss directly
        criterion = weighted_bce_loss
    
    # start logging
    if args.log:
        wandb.init(project = "ShadowMagic Ver 0.1", entity="waterheater", name = name)
        wandb.config = {
          "learning_rate": lr,
          "epochs": epochs, 
          "batch_size": batch_size,
          "crop_size": crop_size,
          "loss":"L1" if l1_loss else "BCE"
        }
        # wandb.watch(net, log_freq=30)

    now = datetime.now()
    dt_formatted = now.strftime("D%Y-%m-%dT%H-%M-%S") 
    if args.name is not None:
        dt_formatted = dt_formatted + "-" + args.name
    model_folder = os.path.join("./checkpoints", dt_formatted)
    os.makedirs(model_folder)

    for epoch in range(epochs):
        # train
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            net.train()
            epoch_loss = 0
            for imgs, lines, gts_list, flat_mask, shade_edge, label in train_loader:
                gts, gts_d2x, gts_d4x, gts_d8x = gts_list
                imgs = imgs.to(device=device, dtype=torch.float32)
                lines = lines.to(device=device, dtype=torch.float32)
                gts = gts.to(device=device, dtype=torch.float32)
                gts_d2x = gts_d2x.to(device=device, dtype=torch.float32)
                gts_d4x = gts_d4x.to(device=device, dtype=torch.float32)
                gts_d8x = gts_d8x.to(device=device, dtype=torch.float32)
                flat_mask = flat_mask.to(device=device, dtype=torch.float32)
                shade_edge = shade_edge.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                # forward
                pred, pred_d2x, pred_d4x, pred_d8x = net(imgs, label)
                
                # compute loss
                loss = 0
                if l1_loss:
                    pred = denormalize(pred)
                    loss = criterion(pred, gts)
                else:
                    loss_bce = criterion(pred, gts)
                    loss = loss + loss_bce
                if ap:
                    loss_ap = anisotropic_penalty(pred, shade_edge)
                    loss = loss + 1e-6 * loss_ap

                # record loss
                epoch_loss += loss.item()
                pbar.set_postfix(**{'Loss:': loss.item()})

                # back propagate
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                # scheduler.step()
                pbar.update(len(imgs))
                
                # record the loss more frequently
                if global_step % 350 == 0 and args.log:
                    wandb.log({'Loss': loss_bce.item()}, step = global_step) 
                    if ap:
                        wandb.log({'Anisotropic Penalty': loss_ap.item()}, step = global_step) 

                # record the image output 
                if global_step % 750 == 0:
                    imgs = denormalize(imgs)
                    if l1_loss:
                        gts = denormalize(gts)
                        pred = denormalize(pred)
                    else:
                        pred = torch.sigmoid(pred)
                    sample = torch.cat((imgs,  pred.repeat(1, 3, 1, 1), 
                                (pred > 0.5).repeat(1, 3, 1, 1), gts.repeat(1, 3, 1, 1)), dim = 0)

                    result_folder = os.path.join("./results/train/", dt_formatted)
                    if os.path.exists(result_folder) is False:
                        logging.info("Creating %s"%str(result_folder))
                        os.makedirs(result_folder)
                    utils.save_image(
                        sample,
                        os.path.join(result_folder, f"{str(global_step).zfill(6)}.png"),
                        nrow=int(imgs.shape[0]),
                        normalize=True,
                        range=(0, 1),
                    )
                    
                    '''let's put the training result on the wandb '''
                    if args.log:
                        fig_res = wandb.Image(np.array(
                            Image.open(os.path.join(result_folder, f"{str(global_step).zfill(6)}.png"))))
                        wandb.log({'Train Result': fig_res}, step = global_step)
                    
                # update the global step
                global_step += 1

        # validation
        if epoch % 50 == 0:
            logging.info('Starting Validation')
            net.eval()
            with torch.no_grad():
                val_counter = 0
                val_figs = []
                dims = None
                for val_img, val_lines, val_gt_list, _, _, label in val_loader:
                    if val_counter > 5: break
                    # predict
                    val_gt, _, _, _ = val_gt_list
                    val_img = val_img.to(device=device, dtype=torch.float32)
                    val_gt = val_gt.to(device=device, dtype=torch.float32)
                    val_lines = val_lines.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)
                    val_pred, _, _, _ = net(val_img, label)
                    if l1_loss:
                        val_gt = denormalize(val_gt)
                        val_pred = denormalize(val_pred)
                    else:
                        val_pred = torch.sigmoid(val_pred)
                    # save result
                    val_img = tensor_to_img(denormalize(val_img))
                    val_pred_2 = tensor_to_img((val_pred > 0.5).repeat(1, 3, 1, 1))
                    val_pred = tensor_to_img(val_pred.repeat(1, 3, 1, 1))
                    val_gt = tensor_to_img(val_gt.repeat(1, 3, 1, 1))
                    val_sample = np.concatenate((val_img, val_pred, val_pred_2, val_gt), axis = 1)
                    if val_counter == 0:
                        dims = (val_sample.shape[1], val_sample.shape[0])
                    else:
                        assert dims is not None
                        val_sample = cv2.resize(val_sample, dims, interpolation = cv2.INTER_AREA)
                    val_counter += 1
                    val_figs.append(val_sample)    
                val_figs = np.concatenate(val_figs, axis = 0)
                Image.fromarray(val_figs).save(os.path.join(result_folder, "%06d_val.png"%global_step))
                if args.log:
                    val_fig_res = wandb.Image(val_figs)
                    wandb.log({"Val Result":val_fig_res}, step = global_step)
        # save model
        if save_cp and epoch % 100 == 0:
            # save trying result in single folder each time
            logging.info('Created checkpoint directory')
            torch.save(net.state_dict(),
                      os.path.join(model_folder, f"CP_epoch{epoch + 1}.pth"))
            logging.info(f'Checkpoint {epoch + 1} saved !')

def tensor_to_img(t):
    return (t.cpu().numpy().squeeze().transpose(1,2,0) * 255).astype(np.uint8)

def get_args():
    parser = argparse.ArgumentParser(description='ShadowMagic Ver 0.2',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-m', '--multi-gpu', action='store_true')
    parser.add_argument('-a', action='store_true', dest='anisotropic_penalty', 
                        default='use anisotropic penalty')
    parser.add_argument('-c', '--crop-size', metavar='C', type=int, default=256,
                        help='the size of random cropping', dest="crop")
    parser.add_argument('-n', '--name', type=str,
                        help='the name for wandb logging', dest="name")
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00005,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-r', '--resize', dest="resize", type=int, default=1024,
                        help='resize the shorter edge of the training image')
    parser.add_argument('-i', '--imgs', dest="imgs", type=str,
                        help='the path to training set')
    parser.add_argument('--log', action="store_true", help='enable wandb log')
    parser.add_argument('--l1', action="store_true", help='use L1 loss instead of BCE loss')
    parser.add_argument('-do', action="store_true", help='enable drop out')

    return parser.parse_args()


if __name__ == '__main__':
    
    __spec__ = None
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')

    net = UNet(in_channels=3, out_channels=1, bilinear=True, l1=args.l1, drop_out = args.do)
    
    if args.multi_gpu:
        logging.info("using data parallel")
        net = nn.DataParallel(net).cuda()
    else:
        net.to(device=device)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    # try:
    #     train_net(
    #                 img_path = args.imgs,
    #                 net = net,
    #                 epochs = args.epochs,
    #                 batch_size = args.batchsize,
    #                 lr = args.lr,
    #                 device = device,
    #                 crop_size = args.crop,
    #                 resize = args.resize,
    #                 l1_loss = args.l1,
    #                 name = args.name,
    #                 drop_out = args.do,
    #                 ap = args.anisotropic_penalty
    #               )

    # # this is interesting, save model when keyborad interrupt
    # except KeyboardInterrupt:
    #     torch.save(net.state_dict(), './checkpoints/INTERRUPTED.pth')
    #     logging.info('Saved interrupt')
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)

    '''for debug'''
    train_net(
                img_path = args.imgs,
                net = net,
                epochs = args.epochs,
                batch_size = args.batchsize,
                lr = args.lr,
                device = device,
                crop_size = args.crop,
                resize = args.resize,
                l1_loss = args.l1,
                name = args.name,
                drop_out = args.do,
                ap = args.anisotropic_penalty
            )

