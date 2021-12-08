import torch
import numpy as np
from torch import nn
import math
import os
import kornia

from torchsummary import summary
from blocks import CBlock_ln
# from einops.layers.torch import Rearrange
# from einops import rearrange
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Local_pred(nn.Module):
    def __init__(self, dim=16, number=3):
        super(Local_pred, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.ReLU()
        # main blocks
        block = CBlock_ln(dim)
        blocks1, blocks2 = [block for _ in range(number)], [block for _ in range(number)]
        self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())


    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1)
        add = self.add_blocks(img1)

        return mul, add


class Global_pred(nn.Module):
    def __init__(self, number=2):
        super(Global_pred, self).__init__()
        self.gamma_base = 1.0  # basic gamma value
        self.color_base = torch.eye(3)  # basic color matrix
        # main blocks
        self.conv_large = nn.Conv2d(3, 32, 7, padding=3, groups=1, stride=4)


    def forward(self, img):
        img = self.conv_large(img)
        #print(img.shape)
        return img

class Dynamic_SID(nn.Module):
    def __init__(self):
        super(Dynamic_SID, self).__init__()
        self.local_net = Local_pred()

    def forward(self, img_low):
        mul, add = self.local_net(img_low)
        img_high = (img_low.mul(mul)).add(add)

        return img_high




if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES']='3'
    local_net = Dynamic_SID().cuda()
    img = torch.Tensor(1, 3, 400, 600)
    summary(local_net, input_size=(3, 400, 600))
    # mul, add = local_net(img)
    # print(mul.shape, add.shape)


