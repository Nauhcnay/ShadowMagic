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
from layers import UNet, Generator, Discriminator
from utils.dataset import BasicDataset
from utils.regions import get_regions
from utils.preprocess import fillmap_to_color
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

def l1_loss_masked(pre, target):
    loss_map = F.l1_loss(pre, target, reduction = 'none')
    mask1 = target < 50 # this might not be helpful
    mask2 = target >= 50
    return (loss_map * mask1 + loss_map * mask2 * 0.1).mean()

def l2_loss_masked(pre, target):
    loss_map = F.l2_loss(pre, target, reduction = 'none')
    mask1 = target < 50 # this might not be helpful
    mask2 = target >= 50
    return (loss_map * mask1 + loss_map * mask2 * 0.1).mean()


def focal_loss_bce(pre, target, flat_mask, gamma = 2):
    
    # compute loss map
    bce_loss = F.binary_cross_entropy_with_logits(pre, target, reduction = 'none')
    
    # create the weight mask from the ground truth
    mask = 1
    mask_pos = (target == 1).float()
    mask_neg = (target == 0).float()
    weights_pos = mask_pos.sum(dim = (2, 3)).unsqueeze(-1).unsqueeze(-1)
    weights_neg = mask_neg.sum(dim = (2, 3)).unsqueeze(-1).unsqueeze(-1)
    
    # let's assume the weight for negative samples are always 1, so the weight for positive samples will adaptively change
    if (weights_pos == 0).all():
        mask = mask * (mask_pos + mask_neg)    
    else:
        mask = mask * (mask_pos * (weights_neg / (weights_pos + 1)) + mask_neg)

    if flat_mask is not None:
        # we increase the weight inside the mask by 10 times, reduce the weight outside the mask by 0.1 times
        mask_pos = flat_mask * 2
        mask_neg = 1 - flat_mask
        mask = mask * (mask_pos + mask_neg)

    ## create the focal loss mask
    pre_scores = torch.sigmoid(pre)
    pre_t = pre_scores * target + (1 - pre_scores) * (1 - target)
    
    ## reduce the weight for very confident prediction results
    bce_loss = bce_loss * mask * ((1 - pre_t) ** gamma)

    return bce_loss.mean()

# seems this indexing based loss doesn't always work well
def weighted_bce_loss(pre, target, flat_mask):
    # compute loss map
    bce_loss = F.binary_cross_entropy_with_logits(pre, target, reduction = 'none')
    # compute loss mask
    weights = [0.01, 1, 1, 0.05, 0.05]
    flat_mask = flat_mask.bool()
    mask_outflat = torch.logical_not(flat_mask)
    mask_pos = torch.logical_and(target.bool(), flat_mask)
    mask_neg = torch.logical_and(torch.logical_not(target.bool()), flat_mask)
    pre_norm = torch.sigmoid(pre)
    mask_FN = torch.logical_and(pre_norm < 0.5, mask_pos) 
    mask_FP = torch.logical_and(pre_norm >= 0.5, mask_neg)
    # masks = [mask_outflat, mask_neg, mask_pos]
    masks = [mask_outflat, mask_neg, mask_pos, mask_FN, mask_FP]

    # apply focal loss
    pre_scores = torch.sigmoid(pre)
    pre_t = pre_scores * target + (1 - pre_scores) * (1 - target)
    bce_loss = bce_loss * ((1 - pre_t) ** 2)

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
              args,
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
              ap = False, 
              aps = 3,
              ckpt = None):

    # create dataloader
    dataset_train = BasicDataset(img_path, crop_size = crop_size, resize = resize, l1_loss = l1_loss or args.l2)
    dataset_val = BasicDataset(img_path, crop_size = crop_size, resize = resize, val = True, l1_loss = l1_loss or args.l2)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=args.worker, pin_memory=False, drop_last=False)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=args.worker, pin_memory=False, drop_last=False)
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
        Loss:            {"L1 or L2" if (l1_loss or args.l2) else "BCE"}
        Attention:       {args.att}
    ''')

    now = datetime.now()
    dt_formatted = now.strftime("D%Y-%m-%dT%H-%M-%S") 
    if name is not None:
        dt_formatted = dt_formatted + "-" + name
    model_folder = os.path.join("./checkpoints", dt_formatted)
    result_folder = os.path.join("./results/train/", dt_formatted)

    # not sure which optimizer will be better
    if args.wgan:
        gen, dis = net
        optimizer_gen = optim.Adam(gen.parameters(), lr=lr, weight_decay=1e-8)
        optimizer_dis = optim.Adam(dis.parameters(), lr=lr, weight_decay=1e-8)
    else:    
        #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    
    if ckpt is not None:
        start_epoch = ckpt['epoch']
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if args.sch:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, last_epoch = start_epoch)
            scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        # model_folder = args.model_folder
        # result_folder = args.result_folder
    else:
        start_epoch = 0
        # args.model_folder = model_folder
        # args.result_folder = result_folder
        if args.sch:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, last_epoch = -1)
    

    # create the loss function
    # the task is in fact a binary classification problem
    if l1_loss:
        criterion = l1_loss_masked
    elif args.l2:
        criterion = l2_loss_masked
    else:
        # let's use the focal loss instead of the BCE loss directly
        if args.base0:
            criterion = focal_loss_bce
        else:
            criterion = weighted_bce_loss
    
    # start logging
    if args.log:
        wandb.init(project = "ShadowMagic Ver 0.2", entity="waterheater", name = name)
        wandb.config = {
          "learning_rate": lr,
          "epochs": epochs, 
          "batch_size": batch_size,
          "crop_size": crop_size,
          "loss":"L1" if l1_loss else "BCE"
        }
        # wandb.watch(net, log_freq=30)

    if os.path.exists(model_folder) == False:
        os.makedirs(model_folder)

    for epoch in range(start_epoch, start_epoch + epochs):
        # train
        pbar = tqdm(train_loader)
        epoch_loss = 0
        # for imgs, lines, gt_list, flat_mask, shade_edge, region, label in pbar:
        for imgs, lines, gts, flat_mask, shade_edge, region, label in pbar:
            # gts, _, _, _ = gts_list
            if args.line_only:
                imgs = lines
            imgs = imgs.to(device=device, dtype=torch.float32)
            gts = gts.to(device=device, dtype=torch.float32)
            # gts_d2x = gts_d2x.to(device=device, dtype=torch.float32)
            # gts_d4x = gts_d4x.to(device=device, dtype=torch.float32)
            # gts_d8x = gts_d8x.to(device=device, dtype=torch.float32)
            flat_mask = flat_mask.to(device=device, dtype=torch.float32)
            shade_edge = shade_edge.to(device=device, dtype=torch.float32)
            region = region.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # forward
            # todo: add wgan branch
            if args.wgan:
                optimizer_dis.zero_grad()
                optimizer_gen.zero_grad()
                # step discriminator first
                gen_fake_nograd = gen(imgs, label).detach()
                dis_real = dis(region, label)
                dis_fake = dis(gen_fake_nograd, label)
                loss_D = -torch.mean(dis_real) + torch.mean(dis_fake)
                loss_D.backward()
                optimizer_dis.step()

                # then step generator second, usually generator is stronger than discriminator
                # but I'm not sure if now there are any other better training strategy
                if global_step % 5 == 0:
                    assert args.l1 or args.l2
                    gen_fake = gen(imgs, label)
                    recon_loss = criterion(fake, region)
                    loss_G = -torch.mean(dis(gen_fake, label))
                    loss_G_all = recon_loss + loss_G
                    loss_G_all.backward()
                    optimizer_gen.step()

            else:
                optimizer.zero_grad()
                pred, _, _, _ = net(imgs, label)
            
                # compute loss
                loss = 0
                if l1_loss or args.l2:
                    loss_l1 = criterion(pred, region)
                    loss += loss_l1
                else:
                    loss_bce = criterion(pred, gts, flat_mask)
                    loss = loss + loss_bce
                if ap and l1_loss == False:
                    loss_ap = anisotropic_penalty(pred, shade_edge, size = aps)
                    loss = loss + 1e-6 * loss_ap

                # record loss
                epoch_loss += loss.item()   
                pbar.set_description("Epoch:%d/%d, Loss:%.4f"%(epoch, start_epoch + epochs, loss.item()))
                # pbar.update(1)

                # back propagate
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                if args.sch:
                    scheduler.step()
                
                # record the loss more frequently
                if global_step % 350 == 0 and args.log:
                    if l1_loss or args.l2:
                        wandb.log({'Loss': loss_l1.item()}, step = global_step) 
                    else:
                        wandb.log({'Loss': loss_bce.item()}, step = global_step) 
                    if ap and l1_loss == False and args.l2 == False:
                        wandb.log({'Anisotropic Penalty': loss_ap.item()}, step = global_step) 

                # record the image output 
                if global_step % 1050 == 0:
                    imgs = denormalize(imgs)
                    if args.line_only:
                        imgs = imgs.repeat((1, 3, 1, 1))
                    if l1_loss or args.l2:
                        gts_ = (denormalize(region) * 255).clamp(0, 255).cpu().numpy()
                        pred_ = (denormalize(pred) * 255).clamp(0, 255).detach().cpu().numpy()
                        gts__ = []
                        pred__ = []
                        for i in range(gts.shape[0]):
                            shad_r_gt, _ = fillmap_to_color(get_regions(gts_[i].squeeze()))
                            shad_r_pre, _ = fillmap_to_color(get_regions(pred_[i].squeeze()))
                            gts__.append((shad_r_gt / 255).transpose((2, 0, 1)))
                            pred__.append((shad_r_pre / 255).transpose((2, 0, 1)))
                        gts_ = torch.Tensor(np.stack(gts__, axis = 0)).to(imgs.device)
                        pred_ = torch.Tensor(np.stack(pred__, axis = 0)).to(imgs.device)
                        gts = denormalize(region).repeat((1,3,1,1))
                        pred = denormalize(pred).repeat((1,3,1,1))
                    else:
                        pred = torch.sigmoid(pred)
                        pred[~flat_mask.bool()] = 0
                        # pred = T.functional.equalize((pred*255).to(torch.uint8)).to(torch.float32) / 255
                        # add advanced filter
                    if l1_loss or args.l2:
                        sample = torch.cat((imgs, gts, pred, gts_, pred_), dim = 0)
                    # elif args.line_only:
                    #     sample = torch.cat((imgs, gts, pred, pred > 0.5), dim = 0)
                    else:
                        sample = torch.cat((imgs, gts.repeat((1, 3, 1, 1)), pred.repeat((1, 3, 1, 1)), 
                                    (pred > 0.5).repeat((1, 3, 1, 1))), dim = 0)

                    
                    if os.path.exists(result_folder) is False:
                        logging.info("Creating %s"%str(result_folder))
                        os.makedirs(result_folder)
                    utils.save_image(
                        sample,
                        os.path.join(result_folder, f"{str(global_step).zfill(6)}.png"),
                        nrow=int(imgs.shape[0]),
                        normalize=True,
                        value_range=(0, 1),
                    )
                    
                    '''let's put the training result on the wandb '''
                    if args.log:
                        fig_res = wandb.Image(np.array(
                            Image.open(os.path.join(result_folder, f"{str(global_step).zfill(6)}.png"))))
                        wandb.log({'Train Result': fig_res}, step = global_step)
                    
            # update the global step
            global_step += 1

        if args.wgan:
            pass
        else:
            # save model for every epoch, but since now the dataset is really small, so we save checkpoint at every 5 epoches
            if epoch % 2 == 0:
                if args.sch:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': scheduler.state_dict(),
                        'param': args
                        },
                      os.path.join(model_folder, "last_epoch.pth"))
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': None,
                        'param': args
                        },
                      os.path.join(model_folder, "last_epoch.pth"))

            # validation
            if epoch % 2 == 0:
                logging.info('Starting Validation')
                net.eval()
                val_bceloss = 0
                val_ap = 0
                with torch.no_grad():
                    val_counter = 0
                    val_figs = []
                    dims = None
                    for val_img, val_lines, val_gt, val_flat_mask, val_shade_edge, val_region, label in val_loader:
                        if args.line_only:
                            val_img = val_lines
                        # predict
                        # val_gt, _, _, _ = val_gt_list
                        val_img = val_img.to(device=device, dtype=torch.float32)
                        val_gt = val_gt.to(device=device, dtype=torch.float32)
                        # val_lines = val_lines.to(device=device, dtype=torch.float32)
                        val_shade_edge = val_shade_edge.to(device=device, dtype=torch.float32)
                        val_flat_mask = val_flat_mask.to(device=device, dtype=torch.float32)
                        val_region = val_region.to(device=device, dtype=torch.float32)
                        label = label.to(device=device, dtype=torch.float32)
                        val_pred, _, _, _ = net(val_img, label)
                        if l1_loss or args.l2:
                            val_bceloss += criterion(val_pred, val_region)
                            val_gt = denormalize(val_region)
                            val_pred = denormalize(val_pred)
                        else:
                            val_pred = torch.sigmoid(val_pred)
                            val_pred[~val_flat_mask.bool()] = 0
                            val_bceloss += criterion(val_pred, val_gt, val_flat_mask)
                            # val_pred = T.functional.equalize((val_pred*255).to(torch.uint8)).to(torch.float32) / 255
                        if ap:
                            val_ap = anisotropic_penalty(val_pred, val_shade_edge, size = aps)
                        # save first 5 prdictions as result
                        if val_counter < 5:
                            if args.line_only:
                                val_img = val_img.repeat((1, 3, 1, 1))
                            val_img = tensor_to_img(denormalize(val_img))
                            if args.l1 or args.l2:
                                val_pred_r, _ = fillmap_to_color(get_regions(tensor_to_img(val_pred)))
                                val_gt_r, _ = fillmap_to_color(get_regions(tensor_to_img(val_gt)))
                                val_pred = tensor_to_img(val_pred.repeat((1, 3, 1, 1)))
                                val_gt = tensor_to_img(val_gt.repeat((1, 3, 1, 1)))
                                val_sample = np.concatenate((val_img, val_pred, val_gt, val_pred_r, val_gt_r), axis = 1).squeeze()
                            else:
                                val_pred_2 = tensor_to_img((val_pred > 0.5).repeat((1, 3, 1, 1)))
                                val_pred = tensor_to_img(val_pred.repeat((1, 3, 1, 1)))
                                val_gt = tensor_to_img(val_gt.repeat((1, 3, 1, 1)))
                                val_sample = np.concatenate((val_img, val_pred, val_pred_2, val_gt), axis = 1).squeeze()

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
                        wandb.log({'Val Loss': (val_bceloss / val_counter)}, step = global_step)
                        if ap and args.l1 == False and args.l2 == False:
                            wandb.log({'Val Anisotropic Penalty': (val_ap / val_counter)}, step = global_step)
            # save model
            if save_cp and epoch % 100 == 0:
                # save trying result in single folder each time
                logging.info('Created checkpoint directory')
                if args.sch:
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': scheduler.state_dict(),
                                'param': args
                                },
                              os.path.join(model_folder, f"CP_epoch{epoch}.pth"))
                else:
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': None,
                                'param': args
                                },
                              os.path.join(model_folder, f"CP_epoch{epoch}.pth"))
                logging.info(f'Checkpoint {epoch} saved !')

def tensor_to_img(t):
    return (t.squeeze(0).cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

def get_args():
    parser = argparse.ArgumentParser(description='ShadowMagic Ver 0.2',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=740,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-m', '--multi-gpu', action='store_true')
    parser.add_argument('-a', action='store_true', dest='anisotropic_penalty', 
                        default='use anisotropic penalty')
    parser.add_argument('-c', '--crop-size', metavar='C', type=int, default=512,
                        help='the size of random cropping', dest="crop")
    parser.add_argument('-aps', '--ap-size', metavar='C', type=int, default=3,
                        help='the size for anisotropic penalty', dest="ap_size")
    parser.add_argument('-n', '--name', type=str,
                        help='the name for wandb logging', dest="name")
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-w', '--worker', metavar='B', type=int, nargs='?', default=0,
                        help='dataloader worker number', dest='worker')
    # 5e-5
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-r', '--resize', dest="resize", type=int, default=1024,
                        help='resize the shorter edge of the training image')
    parser.add_argument('-i', '--imgs', dest="imgs", type=str,
                        help='the path to training set', default = "./dataset")
    parser.add_argument('--log', action="store_true", help='enable wandb log')
    parser.add_argument('--l1', action="store_true", help='use L1 loss instead of BCE loss')
    parser.add_argument('--l2', action="store_true", help='use L2 loss instead of BCE loss')
    parser.add_argument('--do', action="store_true", help='enable drop out')
    parser.add_argument('--att', action="store_true", help='enable attention module')
    parser.add_argument('-sch', action="store_true", help='enable learning rate scheduler')
    parser.add_argument('--line_only', action = 'store_true', help = 'input line drawing instead of line drawing + flat layer')
    parser.add_argument('--base0', action = 'store_true', help = 'switch to the previous focal loss function')
    parser.add_argument('--wgan', action="store_true", help='enable wgan for training')

    return parser.parse_args()


if __name__ == '__main__':
    
    __spec__ = None
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    # load parameters
    if args.load:
        model_load = args.load
        ckpt = torch.load(args.load, map_location='cuda:0')
        args = ckpt['param']
    else:
        ckpt = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')

    if args.wgan:
        gen = Generator(in_channels= 1 if args.line_only else 3, out_channels=1, drop_out = args.do, attention = args.att)
        dis = Discriminator(in_channels = 2 if arg.line_only else 4)
    else:
        net = UNet(in_channels= 1 if args.line_only else 3, out_channels=1, bilinear=True, l1=args.l1, drop_out = args.do, attention = args.att)
    
    # we skip the wgan loading first
    if ckpt is not None:
        net.load_state_dict(ckpt['model_state_dict'])
        logging.info(f'Model loaded from {model_load}')
    
    if args.multi_gpu:
        logging.info("using data parallel")
        net = nn.DataParallel(net).cuda()
    else:
        if args.wgan:
            gen = gen.to(device = device)
            dis = dis.to(device = device)
            net = (gen, dis)
        else:
            net = net.to(device=device)

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
                device = device,
                args = args,
                epochs = args.epochs,
                batch_size = args.batchsize,
                lr = args.lr,
                crop_size = args.crop,
                resize = args.resize,
                l1_loss = args.l1,
                name = args.name,
                drop_out = args.do,
                ap = args.anisotropic_penalty,
                aps = args.ap_size,
                ckpt = ckpt
            )

