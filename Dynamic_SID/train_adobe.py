import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import vgg16

from data_loader_adobe5k import adobe5k_loader
from model import Dynamic_SID

from IQA_pytorch import SSIM, MS_SSIM
from utils import PSNR, adjust_learning_rate, validation, LossNetwork, visualization

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=1)
parser.add_argument('--img_path', type=str, default="/home/czt/DataSets/five5k_dataset/Inputs_jpg")
parser.add_argument('--img_val_path', type=str, default="/home/czt/DataSets/five5k_dataset/UPE_testset/Inputs_jpg")
parser.add_argument('--normalize', action='store_false')    # False

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=0.0001)
# parser.add_argument('--load_pretrain', type=bool, default=False)
parser.add_argument('--pretrain_dir', type=str, default=None)

parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--display_iter', type=int, default=50)
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
train_dataset = adobe5k_loader(images_path=config.img_path, normalize=config.normalize)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)
val_dataset = adobe5k_loader(images_path=config.img_val_path, mode='test', normalize=config.normalize)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

# Loss & Optimizer Setting & Metric
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

# optimizer = torch.optim.Adam([{'params': model.global_net.parameters(),'lr':config.lr*0.1},
#             {'params': model.local_net.parameters(),'lr':config.lr}], lr=config.lr, weight_decay=config.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

device = next(model.parameters()).device
print('the device is:', device)

L1_loss = nn.L1Loss()
L1_smooth_loss = F.smooth_l1_loss

loss_network = LossNetwork(vgg_model)
loss_network.eval()

ssim = SSIM()
psnr = PSNR()
ssim_high = 0
psnr_high = 0

model.train()
print('######## Dynamic SID Time #########')
for epoch in range(config.num_epochs):
    # adjust_learning_rate(optimizer, epoch)
    print('the epoch is:', epoch)
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        # Checking!
        visualization(low_img, 'show/low', iteration)
        visualization(high_img, 'show/high', iteration)
        optimizer.zero_grad()
        model.train()
        _, _, enhance_img = model(low_img)

        #loss = L1_smooth_loss(enhance_img, high_img)+0.04*loss_network(enhance_img, high_img)
        loss = L1_loss(enhance_img, high_img)
        #optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()

        if ((iteration + 1) % config.display_iter) == 0:
            print("Loss at iteration", iteration + 1, ":", loss.item())

    # Evaluation Model
    model.eval()
    PSNR_mean, SSIM_mean = validation(model, val_loader)

    with open(config.log, 'a+') as f:
        f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(SSIM_mean) + 'the PSNR is' + str(PSNR_mean) + '\n')

    if SSIM_mean > ssim_high:
        ssim_high = SSIM_mean
        print('the highest SSIM value is:', str(ssim_high))
        # print(config.snapshots_folder)
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch" + '.pth'))

    f.close()







