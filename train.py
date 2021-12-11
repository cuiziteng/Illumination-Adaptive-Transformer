import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from data_loader import lowlight_loader
from model import Dynamic_SID

from IQA_pytorch import SSIM, MS_SSIM
from utils import PSNR, L_VGG
# from kornia.filters import gaussian_blur2d




# image visualization tools
def visualization(img, img_path, iteration):
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    img = img.cpu().numpy()

    for i in range(img.shape[0]):
        # save name
        name = str(iteration) + '_' + str(i) + '.png'
        print(name)

        img_single = np.transpose(img[i, :, :, :], (1, 2, 0))
        # print(img_single)
        img_single = np.clip(img_single, 0, 1) * 255.0
        img_single = cv2.UMat(img_single).get()
        img_single = img_single / 255.0

        plt.imsave(os.path.join(img_path, name), img_single)


def train(model, train_loader, optimizer, scheduler, config, epoch, L_content, L_vgg, L_ssim):
    # Start Training
    model.train()
    print('The epoch is %d' % (epoch + 1))
    # print('######## Start Training #########')
    # for epoch in range(config.num_epochs):

    for iteration, imgs in enumerate(train_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        # Checking!
        #visualization(low_img, 'show/low', iteration)
        #visualization(high_img, 'show/high', iteration)

        # mul_img, add_img, enhance_img = model(low_img)
        _, _, enhance_img = model(low_img)

        loss_l1 = L_content(enhance_img, high_img)
        #loss_ssim = L_ssim(enhance_img, high_img, as_loss=True)
        #loss_l1 = L_content(enhance_img, high_img) + 0.02*L_vgg(enhance_img, high_img)
        # TODO: perpecl loss , SSIM loss
        # loss_vgg = L_vgg(enhance_img, high_img)

        # loss = loss_l1 + 0.01*loss_vgg

        #loss = loss_l1 + 0.05*loss_ssim
        loss = loss_l1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()

        if ((iteration + 1) % config.display_iter) == 0:
            print("Loss at iteration", iteration + 1, ":", loss.item())


def val(model, val_loader, epoch, ssim, psnr):
    model.eval()

    ssim_list = []
    psnr_list = []
    for i, imgs in enumerate(val_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        _, _, enhanced_img = model(low_img)
        # print(enhanced_img.shape)
        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        #ssim_value = ssim(enhanced_img, high_img).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        # print('The %d image SSIM value is %d:' %(i, ssim_value))
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print('The SSIM Value is:', SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)
    return SSIM_mean, PSNR_mean


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default=1)
    parser.add_argument('--img_path', type=str, default="/home/czt/DataSets/LOL_v2/Train/Low/")
    parser.add_argument('--img_val_path', type=str, default="/home/czt/DataSets/LOL_v2/Test/Low/")
    #parser.add_argument('--img_size', type=tuple, default=(400, 600))
    # parser.add_argument('--mlp_type', type=str, default='resmlp')
    # parser.add_argument('--grad_clip_norm', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    #parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default=None)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_epoch', type=int, default=2)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_folder")
    parser.add_argument('--log', type=str, default="log.txt")

    config = parser.parse_args()
    
    print(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    # Model Setting
    model = Dynamic_SID().cuda()
    if config.pretrain_dir is not None:
        model.load_state_dict(torch.load(config.pretrain_dir))

    # Data Setting
    train_dataset = lowlight_loader(images_path=config.img_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    val_dataset = lowlight_loader(images_path=config.img_val_path, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # Loss & Optimizer Setting
    # optimizer = torch.optim.Adam([{'params': model.color_net.parameters(),'lr':config.lr*0.1},
    #             {'params': model.local_net.parameters(),'lr':config.lr}], lr=config.lr, weight_decay=config.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    device = next(model.parameters()).device
    print('the device is:', device)
    l_vgg = L_VGG(device)
    L_content = nn.L1Loss()
    L_ssim = SSIM()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    ssim = SSIM()
    #ssim = ssim
    psnr = PSNR()
    ssim_high = 0
    psnr_high = 0
    print('######## Start Training #########')
    
    for epoch in range(config.num_epochs):
        train(model, train_loader, optimizer, scheduler, config, epoch, L_content, l_vgg, L_ssim)
        
        ssim_epoch, psnr_epoch = val(model, val_loader, epoch, ssim, psnr)
        with open(config.log, 'a+') as f:
            f.write('epoch'+str(epoch)+':'+'the SSIM is'+str(ssim_epoch)+'the PSNR is'+str(psnr_epoch)+'\n')

        if ssim_epoch > ssim_high:
            ssim_high = ssim_epoch
            print('the highest SSIM value is:', str(ssim_high))
            #print(config.snapshots_folder)
            torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch" + '.pth'))
        f.close()





