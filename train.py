import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from torch import optim
from tqdm import tqdm
from layers import UNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision import utils
from PIL import Image
# let's add the wandb
import wandb


def denormalize(img):
    # denormalize
    return (img / 2 + 0.5).clamp(0, 1)
    return img_np

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
              resize = 1024):

    # create dataloader
    dataset_train = BasicDataset(img_path, crop_size = crop_size, resize = resize)
    dataset_val = BasicDataset(img_path, crop_size = crop_size, resize = resize, val = True)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
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
    ''')

    # not sure which optimizer will be better
    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    
    # create the loss function
    # the task is in fact a binary classification problem
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.L1Loss()

    # start logging
    if args.log:
        wandb.init(project = "ShadowMagic Ver 0.1", entity="waterheater")
        wandb.config = {
          "learning_rate": args.lr,
          "epochs": epochs,
          "batch_size": args.batchsize,
          "crop_size": args.crop
        }
        wandb.watch(net, log_freq=30)

    # start training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for imgs, gts, label in train_loader:

                imgs = imgs.to(device=device, dtype=torch.float32)
                gts = gts.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                # forward
                pred = net(imgs, label)
                
                # '''
                # baseline
                # '''
                # loss1 = criterion(pred, gts)
                
                '''
                weighted loss
                yes, we care more true positive predictions than false positive, we probably need to apply a mask!
                gts can be used as a mask, too
                '''
                # loss of positive labels
                loss1 = criterion(pred * gts, gts) 
                # loss of negative labels
                loss2 = criterion(pred * (1 - gts), torch.zeros(gts.shape).to(device=device, dtype=torch.float32)) 

                # total loss
                loss = 10 * loss1 + 0.5 * loss2

                # record loss
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # back propagate
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                
                global_step += 1
                
                if global_step % 1000 == 0 and args.log:
                # if True and args.log:
                    
                    sample = torch.cat((denormalize(imgs), (pred > 0.5).repeat(1, 3, 1, 1), gts.repeat(1, 3, 1, 1)), dim = 0)
                    if os.path.exists("./results/train/") is False:
                        logging.info("Creating ./results/train/")
                        os.makedirs("./results/train/")

                    utils.save_image(
                        sample,
                        f"./results/train/{str(global_step).zfill(6)}.png",
                        nrow=int(batch_size),
                        normalize=True,
                        range=(0, 1),
                    )
                    
                    '''let's put the training result on the wandb '''
                    if args.log:
                        fig_res = wandb.Image(np.array(Image.open(f"./results/train/{str(global_step).zfill(6)}.png")))
                        wandb.log({'Total Loss': loss.item()}) 
                        wandb.log({'Train Result': fig_res})

                    # let's also run a validation test
                    logging.info('Starting Validation')
                    net.eval()
                    with torch.no_grad():
                        # read validation samples
                        for val_img, val_gt, label in val_loader:
                            # predict
                            val_img = val_img.to(device=device, dtype=torch.float32)
                            val_gt = val_gt.to(device=device, dtype=torch.float32)
                            label = label.to(device=device, dtype=torch.float32)
                            val_pred = net(val_img, label)
                            # save result
                            val_img = tensor_to_img(denormalize(val_img))
                            val_pred = tensor_to_img((val_pred > 0.5).repeat(1, 3, 1, 1))
                            val_gt = tensor_to_img(val_gt.repeat(1, 3, 1, 1))
                            val_sample = np.concatenate((val_img, val_pred, val_gt), axis = 1)
                            val_fig_res = wandb.Image(val_sample)
                            wandb.log({"Val Result":val_fig_res})

        # save model
        if save_cp and epoch % 100 == 0:
            try:
                os.mkdir("checkpoint")
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       f'checkpoint/CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

def tensor_to_img(t):
    return (t.cpu().numpy().squeeze().transpose(1,2,0) * 255).astype(np.uint8)

def get_args():
    parser = argparse.ArgumentParser(description='ShadowMagic Ver 0.1',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=90000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-m', '--multi-gpu', action='store_true')
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

    return parser.parse_args()


if __name__ == '__main__':
    
    __spec__ = None
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(in_channels=3, out_channels=1, bilinear=True)
    
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
                    resize = args.resize
                  )

    # this is interesting, save model when keyborad interrupt
    except KeyboardInterrupt:
        torch.save(net.state_dict(), './checkpoints/INTERRUPTED.pth')
        # logging.info('Saved interrupt')
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