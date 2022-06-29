import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

import os
import argparse
import numpy as np
import random
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

from data_loaders.exposure import exposure_loader
from model.IAT_main import IAT

from IQA_pytorch import SSIM
from utils import PSNR, validation, LossNetwork, visualization, get_dist_info


print(torch.cuda.device_count())
dist.init_process_group(backend='nccl')

parser = argparse.ArgumentParser()
# parser.add_argument('--gpu_id', type=str, default=1)
parser.add_argument("--local_rank", default=-1, type=int, help="")
parser.add_argument('--img_path', type=str, default="/data/unagi0/cui_data/light_dataset/Exposure_CVPR21/train/INPUT_IMAGES")
parser.add_argument('--img_val_path', type=str, default="/data/unagi0/cui_data/light_dataset/Exposure_CVPR21/validation/INPUT_IMAGES")
parser.add_argument("--normalize", action="store_true", help="Default not Normalize in exposure training.")

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=2e-4)   # for batch size 4x2=8
parser.add_argument('--weight_decay', type=float, default=0.0002)
parser.add_argument('--pretrain_dir', type=str, default=None)

parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--display_iter', type=int, default=100)
parser.add_argument('--snapshots_folder', type=str, default="workdirs/snapshots_folder_exposure")

config = parser.parse_args()

print(config)

if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

# Distribute Setting
torch.cuda.set_device(config.local_rank)

# Seed 
seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Model Setting
model = IAT(type='exp').cuda()
if config.pretrain_dir is not None:
    model.load_state_dict(torch.load(config.pretrain_dir))

# Data Setting
train_dataset = exposure_loader(images_path=config.img_path, normalize=config.normalize)
train_sampler = DistributedSampler(train_dataset, drop_last=True, seed=seed)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                           pin_memory=True, sampler=train_sampler)
        
val_dataset = exposure_loader(images_path=config.img_val_path, mode='val', normalize=config.normalize)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

# Loss & Optimizer Setting & Metric
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()

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
rank, _ = get_dist_info()

model.train()
print('######## Start IAT Training #########')
for epoch in range(config.num_epochs):
    # adjust_learning_rate(optimizer, epoch)
    print('the epoch is:', epoch)
    train_sampler.set_epoch(epoch)
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        # Checking!
        #visualization(low_img, 'show/low', iteration)
        #visualization(high_img, 'show/high', iteration)
        optimizer.zero_grad()
        model.train()
        mul, add, enhance_img = model(low_img)
    
        loss = L1_loss(enhance_img, high_img)
        #loss = L1_smooth_loss(enhance_img, high_img)+0.04*loss_network(enhance_img, high_img)
        
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        if ((iteration + 1) % config.display_iter) == 0:
            print("Loss at iteration", iteration + 1, ":", loss.item())
    #print(rank)
    # Evaluation Model
    if rank == 0:
        #print('111')
        model.eval()

        PSNR_mean, SSIM_mean = validation(model, val_loader)

        with open(config.snapshots_folder + '/log.txt', 'a+') as f:
            f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(SSIM_mean) + 'the PSNR is' + str(PSNR_mean) + '\n')

        if SSIM_mean > ssim_high:
            ssim_high = SSIM_mean
            print('the highest SSIM value is:', str(ssim_high))
            torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch" + '.pth'))

        f.close()

    dist.barrier()
