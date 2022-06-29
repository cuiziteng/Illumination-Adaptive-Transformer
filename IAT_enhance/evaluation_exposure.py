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
from model.IAT_main import IAT
from IQA_pytorch import SSIM, MS_SSIM
from data_loaders.exposure import exposure_loader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=3)
parser.add_argument('--img_val_path', type=str, default="/data/unagi0/cui_data/light_dataset/Exposure_CVPR21/test/INPUT_IMAGES/")
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--expert', type=str, default='c')  # Choose the evaluation expert
parser.add_argument('--pre_norm', type=bool, default=False) 
config = parser.parse_args()

print(config)
test_dataset = exposure_loader(images_path=config.img_val_path, mode='test',  expert=config.expert, normalize=config.pre_norm)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

model = IAT(type='exp').cuda()
model.load_state_dict(torch.load("best_Epoch_exposure.pth"))
model.eval()


ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if config.save:
    result_path = config.img_val_path.replace('INPUT_IMAGES', 'Result')
    mkdir(result_path)

with torch.no_grad():
    for i, imgs in tqdm(enumerate(test_loader)):
        #print(i)
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        #print(low_img.shape)
        mul, add ,enhanced_img = model(low_img)
        
        if config.save:
            torchvision.utils.save_image(enhanced_img, result_path + str(i) + '.png')

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)


SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)
