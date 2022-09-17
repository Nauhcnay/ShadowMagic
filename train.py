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


def focal_loss(pre, target, gamma = 5):
    ## create the weight mask from the ground truth
    mask_pos = (target == 1).float()
    mask_neg = (target == 0).float()
    weights_pos = mask_pos.sum(dim = (2, 3)).unsqueeze(-1).unsqueeze(-1)
    weights_neg = mask_neg.sum(dim = (2, 3)).unsqueeze(-1).unsqueeze(-1)
    # let's assume the weight for negative samples are always 1, so the weight for positive samples will adaptively change
    if (weights_pos == 0).all():
        mask_weight = mask_pos + mask_neg    
    else:
        mask_weight = mask_pos * (weights_neg / (weights_pos + 1)) + mask_neg

    ## create the focal loss mask
    pre_scores = torch.sigmoid(pre)
    pre_t = pre_scores * target + (1 - pre_scores) * (1 - target)

    ## compute the bce loss
    bce_loss = F.binary_cross_entropy_with_logits(pre, target, weight = mask_weight, reduction = 'none')

    ## reduce the weight for very confident prediction results
    bce_loss = bce_loss * ((1 - pre_t) ** gamma)

    return bce_loss.mean()

def denormalize(img):
    # denormalize
    return (img / 2 + 0.5).clamp(0, 1)

def train_net(
              img_path,
              net,
              device,
              use_mask = [True, False],
              epochs=999,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              crop_size = 256,
              resize = 1024,
              l1_loss = False):

    # create dataloader
    dataset_train = BasicDataset(img_path, crop_size = crop_size, resize = resize, l1_loss = l1_loss)
    dataset_val = BasicDataset(img_path, crop_size = crop_size, resize = resize, val = True, l1_loss = l1_loss)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    # we don't need valiation currently
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # get dataset lenght
    n_train = len(dataset_train)
    
    # display train summarys
    global_step = 0
    
    mask1_flag, mask2_flag = use_mask
    if mask1_flag and mask2_flag == False:
        mask_str = "Type1"
    elif mask1_flag == False and mask2_flag:
        mask_str = "Type2"
    elif mask1_flag and mask2_flag:
        mask_str = "Both types"
    else:
        mask_str = False

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Crop size:       {crop_size}
        Use Mask:        {mask_str}
        Loss:            {"L1" if l1_loss else "BCE"}
    ''')

    # not sure which optimizer will be better
    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    
    # create the loss function
    # the task is in fact a binary classification problem
    if l1_loss:
        criterion = nn.L1Loss()
    else:
        # let's use the focal loss instead of the BCE loss directly
        criterion = focal_loss
    
    # start logging
    if args.log:
        wandb.init(project = "ShadowMagic Ver 0.1", entity="waterheater")
        wandb.config = {
          "learning_rate": lr,
          "epochs": epochs, 
          "batch_size": batch_size,
          "crop_size": crop_size,
          "use_mask":use_mask,
          "loss":"L1" if l1_loss else "BCE"
        }
        wandb.watch(net, log_freq=30)

    now = datetime.now()
    dt_formatted = now.strftime("D%Y-%m-%dT%H-%M-%S")
    model_folder = os.path.join("./checkpoints", dt_formatted)
    os.makedirs(model_folder)

    # start training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for imgs, gts, mask, label in train_loader:

                imgs = imgs.to(device=device, dtype=torch.float32)
                gts = gts.to(device=device, dtype=torch.float32)
                mask = mask.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                # forward
                pred = net(imgs, label)
                
                if l1_loss:
                    pred = denormalize(pred)

                if mask1_flag == False and mask2_flag == False:
                    # '''
                    # baseline
                    # '''
                    loss = criterion(pred, gts)
                else:
                    '''
                    weighted loss
                    we only care the flat regions shadow, so we could ignore the false positive prediction at the background
                    '''
                    if mask1_flag and mask2_flag == False:
                        mask1 = mask
                        # loss of labels outside flat mask
                        loss1 = criterion(pred * (1 - mask1), gts * (1 - mask1)) 
                        # loss of labels inside flat mask
                        loss2 = criterion(pred * mask1, gts * mask1)
                        loss3 = 0
                    if mask2_flag and mask1_flag == False:
                        if l1_loss:
                            mask2 = denormalize(gts)
                        else:
                            mask2 = gts
                        # loss that outside gt mask
                        loss1 = criterion(pred * (1 - mask2), gts * (1 - mask2)) 
                        # loss that inside of gt mask
                        loss2 = criterion(pred * mask2, gts * mask2)
                        loss3 = 0
                    if mask1_flag and mask2_flag:
                        mask1 = mask
                        if l1_loss:
                            mask2 = denormalize(gts)
                        else:
                            mask2 = gts
                        mask3 = mask1 * (1 - mask2)
                        # loss that outside the flat mask
                        loss1 = criterion(pred * (1 - mask1), gts * (1 - mask1)) 
                        # loss that inside gt mask
                        loss2 = criterion(pred * mask2, gts * mask2)
                        # loss that inside flat mask and negtive label of gt mask
                        loss3 = criterion(pred * mask3, gts * mask3)
                    # total loss
                    loss = 0.5 * loss1 + loss2 + 1.5 * loss3

                # record loss
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # back propagate
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                
                # record the loss more frequently
                if global_step % 20 == 0 and args.log:
                    wandb.log({'Total Loss': loss.item()}) 
                    if mask1_flag or mask2_flag:
                        wandb.log({'Loss outside flat/gt mask': loss1.item()}) 
                        wandb.log({'Loss inside flat/flat only/gt mask)': loss2.item()}) 
                        wandb.log({'Loss inside gt mask (if loss3 is none zero)': loss3.item()}) 
                    # wandb.log({'Loss pos labels inside GT': loss3.item()}) 

                # record the image output 
                # if True:
                if global_step % 350 == 0:
                    imgs = denormalize(imgs)
                    if l1_loss:
                        gts = denormalize(gts)
                        pred = denormalize(pred)
                    else:
                        pred = torch.sigmoid(pred)
                    sample = torch.cat((imgs,  pred.repeat(1, 3, 1, 1),
                        (pred > 0.9).repeat(1, 3, 1, 1), (pred > 0.5).repeat(1, 3, 1, 1),
                        gts.repeat(1, 3, 1, 1)), dim = 0)
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
                        wandb.log({'Total Loss': loss.item()}) 
                        wandb.log({'Train Result': fig_res})

                        # let's also run a validation test
                        logging.info('Starting Validation')
                        net.eval()
                        with torch.no_grad():
                            # read validation samples
                            val_counter = 0
                            val_figs = []
                            dims = None
                            for val_img, val_gt, val_mask, label in val_loader:
                                if val_counter > 5: break
                                # predict
                                val_img = val_img.to(device=device, dtype=torch.float32)
                                val_gt = val_gt.to(device=device, dtype=torch.float32)
                                label = label.to(device=device, dtype=torch.float32)
                                val_pred = net(val_img, label)
                                val_pred = torch.sigmoid(val_pred)
                                if l1_loss:
                                    val_gt = denormalize(val_gt)
                                # save result
                                val_img = tensor_to_img(denormalize(val_img))
                                val_pred_1 = tensor_to_img((val_pred > 0.9).repeat(1, 3, 1, 1))
                                val_pred_2 = tensor_to_img((val_pred > 0.5).repeat(1, 3, 1, 1))
                                val_pred = tensor_to_img(val_pred.repeat(1, 3, 1, 1))
                                val_gt = tensor_to_img(val_gt.repeat(1, 3, 1, 1))
                                val_sample = np.concatenate((val_img, val_pred, val_pred_1, val_pred_2, val_gt), axis = 1)
                                if val_counter == 0:
                                    dims = (val_sample.shape[1], val_sample.shape[0])
                                else:
                                    assert dims is not None
                                    val_sample = cv2.resize(val_sample, dims, interpolation = cv2.INTER_AREA)
                                val_figs.append(val_sample)    
                            val_figs = np.concatenate(val_figs, axis = 1)
                            val_fig_res = wandb.Image(val_figs)
                            wandb.log({"Val Result":val_fig_res})

                # update the global step
                global_step += 1
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
    parser = argparse.ArgumentParser(description='ShadowMagic Ver 0.1',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=90000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-m', '--multi-gpu', action='store_true')
    parser.add_argument('-w1', '--weighted-loss1', action='store_true', dest="mask1",
                        help="use mask type 1 to weight the loss computation")
    parser.add_argument('-w2', '--weighted-loss2', action='store_true', dest="mask2",
                        help="use mask type 2 to weight the loss computation")
    parser.add_argument('-c', '--crop-size', metavar='C', type=int, default=512,
                        help='the size of random cropping', dest="crop")
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-r', '--resize', dest="resize", type=int, default=1024,
                        help='resize the shorter edge of the training image')
    parser.add_argument('-i', '--imgs', dest="imgs", type=str,
                        help='the path to training set')
    parser.add_argument('--log', action="store_true", help='enable wandb log')
    parser.add_argument('--l1', action="store_true", help='use L1 loss instead of BCE loss')

    return parser.parse_args()


if __name__ == '__main__':
    
    __spec__ = None
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')

    net = UNet(in_channels=3, out_channels=1, bilinear=True, l1=args.l1)
    
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

    try:
        train_net(
                    img_path = args.imgs,
                    net = net,
                    epochs = args.epochs,
                    batch_size = args.batchsize,
                    lr = args.lr,
                    device = device,
                    crop_size = args.crop,
                    resize = args.resize,
                    use_mask = [args.mask1, args.mask2],
                    l1_loss = args.l1
                  )

    # this is interesting, save model when keyborad interrupt
    except KeyboardInterrupt:
        torch.save(net.state_dict(), './checkpoints/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    '''for debug'''
    # train_net(
    #             img_path = args.imgs,
    #             net = net,
    #             epochs = args.epochs,
    #             batch_size = args.batchsize,
    #             lr = args.lr,
    #             device = device,
    #             crop_size = args.crop,
    #             resize = args.resize
    #         )

