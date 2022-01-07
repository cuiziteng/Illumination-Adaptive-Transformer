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
from utils import visualization
# Code change from "https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/dataloader.py"
# By Ziteng Cui, cuiziteng@sjtu.edu.cn
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
        self.train_resize = True
        self.test_resize = True
        print("Total examples:", len(self.data_list))
        #print("Total testing examples:", len(self.test_list))
        # self.transform_train = transforms.Compose()

    # # Data Augmentation
    # # TODO: more data augmentation methods
    # def FLIP_LR(self, low, high):
    #     if random.random() > 0.5:
    #         low = low.transpose(Image.FLIP_LEFT_RIGHT)
    #         high = high.transpose(Image.FLIP_LEFT_RIGHT)
    #     return low, high
    #
    # def FLIP_UD(self, low, high):
    #     if random.random() > 0.5:
    #         low = low.transpose(Image.FLIP_TOP_BOTTOM)
    #         high = high.transpose(Image.FLIP_TOP_BOTTOM)
    #     return low, high
    #
    # def get_params(self, low):
    #     self.w, self.h = low.size
    #     #self.crop_height = random.randint(self.h / 2, self.h)  # random.randint(self.MinCropHeight, self.MaxCropHeight)
    #     #self.crop_width = random.randint(self.w / 2, self.w)  # random.randint(self.MinCropWidth,self.MaxCropWidth)
    #     self.crop_height = 512 #random.randint(self.MinCropHeight, self.MaxCropHeight)
    #     self.crop_width = 512 #random.randint(self.MinCropWidth,self.MaxCropWidth)
    #
    #     i = random.randint(0, self.h - self.crop_height)
    #     j = random.randint(0, self.w - self.crop_width)
    #     return i, j
    #
    # def Random_Crop(self, low, high):
    #     self.i, self.j = self.get_params((low))
    #     #if random.random() > 0.5:
    #     low = low.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
    #     high = high.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
    #     return low, high

    def __getitem__(self, index):
        img_id = self.data_list[index]

        if self.mode == 'train':

            data_lowlight = Image.open(osp.join(self.low_path, img_id))
            data_highlight = Image.open(osp.join(self.high_path, img_id))


            #print('000', data_lowlight.size)

            if data_lowlight.size[0] <= data_lowlight.size[1]:
                #print('True')
                data_lowlight = data_lowlight.transpose(Image.ROTATE_90)
                data_highlight = data_highlight.transpose(Image.ROTATE_90)

            #print('111', data_lowlight.size)

            if self.train_resize:    # Same as 'https://github.com/zzyfd/STAR-pytorch'.
                data_lowlight = data_lowlight.resize((1200, 900), Image.ANTIALIAS)
                data_highlight = data_highlight.resize((1200, 900), Image.ANTIALIAS)
            #print('222', data_lowlight.size)
            data_lowlight, data_highlight = self.FLIP_LR(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.FLIP_UD(data_lowlight, data_highlight)
            #data_lowlight, data_highlight = self.Random_Crop(data_lowlight, data_highlight)
            #print('333', data_lowlight.size)

            # print(self.w, self.h)
            # print(data_lowlight.size, data_highlight.size)

            data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)

            if self.normalize:
                # data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
                transform_input = Compose(
                    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
                transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
                # return transform_input(data_lowlight).permute(2, 0, 1), transform_gt(data_highlight).permute(2, 0, 1)
                return transform_input(data_lowlight), transform_gt(data_highlight)
            else:
                data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(
                    data_highlight).float()
                #print(data_lowlight.shape, data_highlight.shape)
                return data_lowlight.permute(2, 0, 1), data_highlight.permute(2, 0, 1)

        elif self.mode == 'test':
            data_lowlight = Image.open(osp.join(self.low_path, img_id))
            data_highlight = Image.open(osp.join(self.high_path, img_id))

            data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)

            if data_lowlight.shape[0] >= data_lowlight.shape[1]:
                data_lowlight = cv2.transpose(data_lowlight)
                data_highlight = cv2.transpose(data_highlight)
            #print(data_lowlight.shape)
            if self.test_resize:    # Same as 'https://github.com/zzyfd/STAR-pytorch'.
                data_lowlight = cv2.resize(data_lowlight, (1200, 900))
                data_highlight = cv2.resize(data_highlight, (1200, 900))
            #print(data_lowlight.shape)
            if self.normalize:
                # data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
                transform_input = Compose(
                    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
                transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
                # return transform_input(data_lowlight).permute(2, 0, 1), transform_gt(data_highlight).permute(2, 0, 1)
                return transform_input(data_lowlight), transform_gt(data_highlight)
            else:
                data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(
                    data_highlight).float()
                return data_lowlight.permute(2, 0, 1), data_highlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train_path = '/home/czt/DataSets/five5k_dataset/Inputs_jpg'
    test_path = '/home/czt/DataSets/five5k_dataset/UPE_testset/Inputs_jpg'
    test_dataset = adobe5k_loader(test_path, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1,
                                               pin_memory=True)
    for iteration, imgs in enumerate(test_loader):
        print(iteration)
        print(imgs[0].shape)
        print(imgs[1].shape)
        low_img = imgs[0]
        high_img = imgs[1]
        visualization(low_img, 'show/low', iteration)
        visualization(high_img, 'show/high', iteration)
