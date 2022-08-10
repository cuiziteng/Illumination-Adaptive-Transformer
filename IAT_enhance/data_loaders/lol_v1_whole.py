import os
import os.path as osp

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
from glob import glob
import random
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype
import torchvision.transforms.functional as TF

random.seed(1143)

def populate_train_list(images_path, mode='train'):
    # print(images_path)
    image_list_lowlight = glob(images_path + '*.png')
    train_list = image_list_lowlight
    if mode == 'train':
        random.shuffle(train_list)

    return train_list

class lowlight_loader_new(data.Dataset):

    def __init__(self, images_path, mode='train'):
        self.train_list = populate_train_list(images_path, mode)
        self.mode = mode
        self.data_list = self.train_list
        print("Total examples:", len(self.train_list))
    

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        #ps = 256 # Training Patch Size 
        if self.mode == 'train':
            data_lowlight = Image.open(data_lowlight_path).convert('RGB')
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high')).convert('RGB')
            w, h = data_lowlight.size
            data_lowlight = TF.to_tensor(data_lowlight)
            data_highlight = TF.to_tensor(data_highlight)
            hh, ww = data_highlight.shape[1], data_highlight.shape[2]

            # rr = random.randint(0, hh - ps)
            # cc = random.randint(0, ww - ps)
            aug = random.randint(0, 3)

            # Crop patch
            # data_lowlight = data_lowlight[:, rr:rr + ps, cc:cc + ps]
            # data_highlight = data_highlight[:, rr:rr + ps, cc:cc + ps]

            # Data Augmentations
            if aug == 1:
                data_lowlight = data_lowlight.flip(1)
                data_highlight = data_highlight.flip(1)
            elif aug == 2:
                data_lowlight = data_lowlight.flip(2)
                data_highlight = data_highlight.flip(2)
            # elif aug == 3:
            #     data_lowlight = torch.rot90(data_lowlight, dims=(1, 2))
            #     data_highlight = torch.rot90(data_highlight, dims=(1, 2))
            # elif aug == 4:
            #     data_lowlight = torch.rot90(data_lowlight, dims=(1, 2), k=2)
            #     data_highlight = torch.rot90(data_highlight, dims=(1, 2), k=2)
            # elif aug == 5:
            #     data_lowlight = torch.rot90(data_lowlight, dims=(1, 2), k=3)
            #     data_highlight = torch.rot90(data_highlight, dims=(1, 2), k=3)
            # elif aug == 6:
            #     data_lowlight = torch.rot90(data_lowlight.flip(1), dims=(1, 2))
            #     data_highlight = torch.rot90(data_highlight.flip(1), dims=(1, 2))
            # elif aug == 7:
            #     data_lowlight = torch.rot90(data_lowlight.flip(2), dims=(1, 2))
            #     data_highlight = torch.rot90(data_highlight.flip(2), dims=(1, 2))

            filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]

            return data_lowlight, data_highlight, filename

        elif self.mode == 'val':
            data_lowlight = Image.open(data_lowlight_path).convert('RGB')
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high')).convert('RGB')
            # Validate on center crop

            data_lowlight = TF.to_tensor(data_lowlight)
            data_highlight = TF.to_tensor(data_highlight)

            filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]

            return data_lowlight, data_highlight, filename

        elif self.mode == 'test':
            data_lowlight = Image.open(data_lowlight_path).convert('RGB')
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high')).convert('RGB')
            
            data_lowlight = TF.to_tensor(data_lowlight)
            data_highlight = TF.to_tensor(data_highlight)

            filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]
            #print(filename)
            return data_lowlight, data_highlight, filename
            
    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    images_path = '/data/unagi0/cui_data/light_dataset/LOL_v1/eval15/low/'

    train_list = populate_train_list(images_path)
    #print(train_list)
    train_dataset = lowlight_loader_new(images_path, mode='val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4,
                                               pin_memory=True)
    for iteration, imgs in enumerate(train_loader):
        #print(iteration)
        print(imgs[0].shape)
        print(imgs[1].shape)
        print(imgs[2][0])
        # print(imgs[0])
