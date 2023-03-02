import os 
import torch
import cv2
import argparse
import warnings
import torchvision
import numpy as np
from utils import PSNR, validation, LossNetwork
from model.IAT_main import IAT
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='demo_imgs/low_demo.jpg')
parser.add_argument('--normalize', type=bool, default=False)
parser.add_argument('--task', type=str, default='enhance', help='Choose from exposure or enhance')
config = parser.parse_args()

# Weights path
exposure_pretrain = r'best_Epoch_exposure.pth'
enhance_pretrain = r'best_Epoch_lol_v1.pth'

normalize_process = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

## Load Pre-train Weights
model = IAT().cuda()
if config.task == 'exposure':
    model.load_state_dict(torch.load(exposure_pretrain))
elif config.task == 'enhance':
    model.load_state_dict(torch.load(enhance_pretrain))
else:
    warnings.warn('Only could be exposure or enhance')
model.eval()


## Load Image
img = Image.open(config.file_name)
img = (np.asarray(img)/ 255.0)
if img.shape[2] == 4:
    img = img[:,:,:3]
input = torch.from_numpy(img).float().cuda()
input = input.permute(2,0,1).unsqueeze(0)
if config.normalize:    # False
    input = normalize_process(input)

## Forward Network
_, _ ,enhanced_img = model(input)

torchvision.utils.save_image(enhanced_img, 'result.png')
