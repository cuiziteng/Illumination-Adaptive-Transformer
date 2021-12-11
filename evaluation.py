import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time

from model import Dynamic_SID
from IQA_pytorch import SSIM, MS_SSIM
from utils import PSNR

ssim = SSIM()
# ms_ssim = MS_SSIM()
psnr = PSNR()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def test(image_path, model_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    #print(data_lowlight.shape)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    if True:
        data_highlight = Image.open(image_path.replace('low', 'normal').replace('Low','Normal'))
        data_highlight = (np.asarray(data_highlight) / 255.0)
        data_highlight = torch.from_numpy(data_highlight).float()
        data_highlight = data_highlight.permute(2, 0, 1)
        data_highlight = data_highlight.cuda().unsqueeze(0)

    print('low', data_lowlight.shape)
    print('high', data_highlight.shape)
    model = Dynamic_SID()
    model.eval()
    model = model.cuda()
    # model.eval()
    model.load_state_dict(torch.load(model_path))

    start = time.time()
    mul_image, add_image, enhanced_image = model(data_lowlight)
    end_time = (time.time() - start)
    print('The consuming time is:', end_time, 's')
    # print(add_image)
    ssim_value = ssim(enhanced_image, data_highlight, as_loss=False)
    #ssim_value = ssim(enhanced_image, data_highlight)
    psnr_value = psnr(enhanced_image, data_highlight)
    # print(enhanced_img.shape)
    # result_path = image_path.replace('low', 'result')
    # mul_path = image_path.replace('low', 'mul')
    # add_path = image_path.replace('low', 'add')

    # torchvision.utils.save_image(mul_image, mul_path)
    # torchvision.utils.save_image(add_image, add_path)
    # torchvision.utils.save_image(enhanced_image, result_path)
    return ssim_value.item(), psnr_value.item()

if __name__ == '__main__':
    
    filePath = '/home/czt/DataSets/LOL_v2/Test/Low'
    # result_path = filePath.replace('low', 'result')
    # mul_path = filePath.replace('low', 'mul')
    # add_path = filePath.replace('low', 'add')
    # mkdir(result_path)
    # mkdir(mul_path)
    # mkdir(add_path)

# test_images
    with torch.no_grad():
        model_path = '/home/czt/Low_light_MLP/shortcut/best_Epoch.pth'
        ssim_list,psnr_list = [], []
        for file in os.listdir(filePath):
            print('0000',file)
            image_path = os.path.join(filePath, file)
            print(image_path)
            ssim_value, psnr_value = test(image_path, model_path)
            ssim_list.append(ssim_value)
            psnr_list.append(psnr_value)
            print(psnr_list)

        SSIM_mean, PSNR_mean = np.mean(ssim_list), np.mean(psnr_list)
        print('The SSIM Value is:', SSIM_mean)
        print('The PSNR Value is:', PSNR_mean)