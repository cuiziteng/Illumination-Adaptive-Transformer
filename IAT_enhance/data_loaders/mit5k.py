import os
import os.path as osp
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image, ImageOps
import glob
import random
import cv2
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype
from glob import glob
# By Ziteng Cui, cui@mi.t.u-tokyo.ac.jp
random.seed(1143)


# input: low light image path
# return: train image ids, test image ids

def populate_train_list(images_path, mode='train'):

    train_list = [os.path.basename(f) for f in glob(os.path.join(images_path, '*.jpg'))]
    train_list.sort()

    if mode == 'train':
        random.shuffle(train_list)

    return train_list


class adobe5k_loader(data.Dataset):

    def __init__(self, images_path, mode='train', normalize=False):
        self.train_list = populate_train_list(images_path, mode)
        # self.h, self.w = int(img_size[0]), int(img_size[1])
        self.mode = mode    # train or test
        self.data_list = self.train_list
        self.low_path = images_path
        self.high_path = images_path.replace('Inputs_jpg', 'Experts_C')
        self.normalize = normalize
        self.resize = True
        #self.image_size = 1200
        #self.image_size_w = 900
        self.image_size = 600
        self.image_size_w = 450
        #self.test_resize = True
        print("Total examples:", len(self.data_list))
        #print("Total testing examples:", len(self.test_list))
        # self.transform_train = transforms.Compose()

    def FLIP_aug(self, low, high):
        if random.random() > 0.5:
            low = cv2.flip(low, 0)
            high = cv2.flip(high, 0)

        if random.random() > 0.5:
            low = cv2.flip(low, 1)
            high = cv2.flip(high, 1)

        return low, high


    def get_params(self, low):
        self.h, self.w = low.shape[0], low.shape[1] # 900, 1200
        #print(self.h, self.w)
        #self.crop_height = random.randint(self.h / 2, self.h)  # random.randint(self.MinCropHeight, self.MaxCropHeight)
        #self.crop_width = random.randint(self.w / 2, self.w)  # random.randint(self.MinCropWidth,self.MaxCropWidth)
        self.crop_height = self.h / 2 #random.randint(self.MinCropHeight, self.MaxCropHeight)
        self.crop_width = self.w / 2 #random.randint(self.MinCropWidth,self.MaxCropWidth)

        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return i, j

    def Random_Crop(self, low, high):
        self.i, self.j = self.get_params(low)
        self.i, self.j = int(self.i), int(self.j)
        #if random.random() > 0.5:
        low = low[self.i: self.i + int(self.crop_height), self.j: self.j + int(self.crop_width)]
        high = high[self.i: self.i + int(self.crop_height), self.j: self.j + int(self.crop_width)]
        return low, high

    def __getitem__(self, index):
        img_id = self.data_list[index]
        
        #data_lowlight = Image.open(osp.join(self.low_path, img_id))
        data_lowlight = cv2.imread(osp.join(self.low_path, img_id), cv2.IMREAD_UNCHANGED)
        data_highlight = cv2.imread(osp.join(self.high_path, img_id), cv2.IMREAD_UNCHANGED)

        if data_lowlight.shape[0] >= data_lowlight.shape[1]:
            data_lowlight = cv2.transpose(data_lowlight)
            data_highlight = cv2.transpose(data_highlight)

        if self.resize:
            data_lowlight = cv2.resize(data_lowlight, (self.image_size, self.image_size_w))
            data_highlight = cv2.resize(data_highlight, (self.image_size, self.image_size_w))
        #print(data_lowlight.shape)
        if self.mode == 'train':    #data augmentation
            data_lowlight, data_highlight = self.FLIP_aug(data_lowlight, data_highlight)
            #data_lowlight, data_highlight = self.Random_Crop(data_lowlight, data_highlight)
        #print(data_lowlight.shape)
        data_lowlight = (np.asarray(data_lowlight[..., ::-1]) / 255.0)
        data_highlight = (np.asarray(data_highlight[..., ::-1]) / 255.0)

        data_lowlight = torch.from_numpy(data_lowlight).float()  # float32
        data_highlight = torch.from_numpy(data_highlight).float()  # float32

        return data_lowlight.permute(2, 0, 1), data_highlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train_path = '/home/czt/DataSets/five5k_dataset/Inputs_jpg'
    test_path = '/home/czt/DataSets/five5k_dataset/UPE_testset/Inputs_jpg'
    test_dataset = adobe5k_loader(train_path, mode='train')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1,
                                               pin_memory=True)
    for iteration, imgs in enumerate(test_loader):
        print(iteration)
        print(imgs[0].shape)
        print(imgs[1].shape)
        low_img = imgs[0]
        high_img = imgs[1]
        # visualization(low_img, 'show/low', iteration)
        # visualization(high_img, 'show/high', iteration)