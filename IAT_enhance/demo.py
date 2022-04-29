import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype
import torch.optim
import torch.nn.functional as F

import argparse
from PIL import Image
import time
import os
import numpy as np
from model import Dynamic_SID

parser = argparse.ArgumentParser()
parser.add_argument('--input_img', type=str)
parser.add_argument('--output_img', type=str)
parser.add_argument('--model_path', type=str)
config = parser.parse_args()
print(config)

os.environ['CUDA_VISIBLE_DEVICES']='1'
pre_norm = True
## Setting Image Path
image_path = config.input_img
data_lowlight = Image.open(image_path).convert('RGB')
data_lowlight = (np.asarray(data_lowlight)/255.0)

if pre_norm:
    print('normalizing...')
    transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
    data_lowlight = transform_input(data_lowlight)
else:
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)

print(data_lowlight.shape)
data_lowlight = data_lowlight.cuda().unsqueeze(0)

## Setting Model
model = Dynamic_SID().cuda()
model.load_state_dict(torch.load(config.model_path))
model.eval()

## Testing
start = time.time()
_, _ ,enhanced_img = model(data_lowlight)

end_time = (time.time() - start)
print(end_time)

## Saving
result_path = config.output_img
if not os.path.exists(result_path.replace('/'+image_path.split("/")[-1],'')):
    os.makedirs(result_path.replace('/'+image_path.split("/")[-1],''))

torchvision.utils.save_image(enhanced_img, result_path)