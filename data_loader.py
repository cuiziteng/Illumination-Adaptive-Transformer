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

# Code change from "https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/dataloader.py"
# By Ziteng Cui, cuiziteng@sjtu.edu.cn
random.seed(1143)


def populate_train_list(images_path):
    # print(images_path)
    image_list_lowlight = glob.glob(images_path + '*.png')
    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list


# Data Augmentation
# TODO: more data augmentation methods
def FLIP_LR(input1, input2):
    if random.random() > 0.5:
        input1 = input1.transpose(Image.FLIP_LEFT_RIGHT)
        input2 = input2.transpose(Image.FLIP_LEFT_RIGHT)
    return input1, input2


def FLIP_UD(input1, input2):
    if random.random() > 0.5:
        input1 = input1.transpose(Image.FLIP_TOP_BOTTOM)
        input2 = input2.transpose(Image.FLIP_TOP_BOTTOM)
    return input1, input2


class lowlight_loader(data.Dataset):

    def __init__(self, images_path, img_size, mode='train'):
        self.train_list = populate_train_list(images_path)
        self.h, self.w = int(img_size[0]), int(img_size[1])
        # train or test
        self.mode = mode
        self.data_list = self.train_list
        print("Total examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        # Low light path
        data_lowlight = Image.open(data_lowlight_path)
        # High light path
        data_highlight = Image.open(data_lowlight_path.replace('low', 'high'))

        # data augmentation
        # TODO: add light change augmentation (i.e. gamma, RetiNex)
        if self.mode == 'train':
            data_lowlight, data_highlight = FLIP_LR(data_lowlight, data_highlight)
            data_lowlight, data_highlight = FLIP_UD(data_lowlight, data_highlight)

            data_lowlight = data_lowlight.resize((self.w, self.h), Image.ANTIALIAS)
            data_highlight = data_highlight.resize((self.w, self.h), Image.ANTIALIAS)

        data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)

        data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(
            data_highlight).float()

        return data_lowlight.permute(2, 0, 1), data_highlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    images_path = '/home/czt/DataSets/LOL_dataset/our485/low/'
    train_dataset = lowlight_loader(images_path, (400, 600))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4,
                                               pin_memory=True)
    for iteration, imgs in enumerate(train_loader):
        print(iteration)
        print(imgs[0].shape)
        print(imgs[1].shape)
