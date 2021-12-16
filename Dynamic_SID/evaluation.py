import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import argparse
import numpy as np
from utils import PSNR, validation, LossNetwork
from model import Dynamic_SID
from IQA_pytorch import SSIM, MS_SSIM
from data_loader import lowlight_loader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=2)
parser.add_argument('--img_val_path', type=str, default="/home/czt/DataSets/LOL_v2/Test/Low/")
parser.add_argument('--model_path', type=str, default="/home/czt/Dynamic_SID/snapshots_folder/local_new150/best_Epoch.pth")
parser.add_argument('--normalize', type=bool, default=False, help='do the pre-normalization or not')
config = parser.parse_args()

print(config)

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

model = Dynamic_SID().cuda()
#model = nn.DataParallel(model)
model.load_state_dict(torch.load("/home/czt/Dynamic_SID/snapshots_folder/best_Epoch.pth"))
model.eval()
val_dataset = lowlight_loader(images_path=config.img_val_path, mode='test', normalize=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []

# model.local_net.eval()
# model.global_net.eval()

with torch.no_grad():
    for i, imgs in enumerate(val_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        #print(low_img.shape)
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
