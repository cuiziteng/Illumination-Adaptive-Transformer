import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math
import os
import kornia

from torchsummary import summary
from blocks import CBlock_ln, SwinTransformerBlock, Aff
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from einops.layers.torch import Rearrange
from einops import rearrange
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# drop_rate = 

class Local_pred(nn.Module):
    def __init__(self, dim=16, number=4, type='ccc'):
        super(Local_pred, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type =='ccc':  
            #blocks1, blocks2 = [block for _ in range(number)], [block for _ in range(number)]
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type =='ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type =='cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        #self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1)
        add = self.add_blocks(img1)

        return mul, add

# Dense Connection
class Local_pred_new(nn.Module):
    def __init__(self, dim=16, number=3, type='ccc'):
        super(Local_pred_new, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.PReLU()
        # main blocks
        self.mul_block1 = CBlock_ln(dim, drop_path=0.01)
        self.mul_block2 = CBlock_ln(dim, drop_path=0.05)
        self.mul_block3 = CBlock_ln(dim, drop_path=0.1)

        self.add_block1 = CBlock_ln(dim, drop_path=0.01)
        self.add_block2 = CBlock_ln(dim, drop_path=0.05)
        self.add_block3 = CBlock_ln(dim, drop_path=0.1)
        
        self.mul_head = nn.Sequential(nn.Conv2d(dim*3, dim, 1), nn.PReLU())
        self.add_head = nn.Sequential(nn.Conv2d(dim*3, dim, 1), nn.PReLU())
        # self.add_end = nn.Sequential(nn.Conv2d(22, 3, 3, 1, 1))
        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        #self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, img):
        fea = self.relu(self.conv1(img)) 

        mul1 = self.mul_block1(fea)
        mul2 = self.mul_block2(mul1)
        mul3 = self.mul_block3(mul2)
        mul = self.mul_head(torch.cat([mul1, mul2, mul3], 1))
        mul += fea
        mul = self.mul_end(mul)

        add1 = self.add_block1(fea)
        add2 = self.add_block2(add1)
        add3 = self.add_block3(add2)
        add = self.add_head(torch.cat([add1, add2, add3], 1))
        #add += fea
        add = self.add_end(add)

        #print(mul.shape, add.shape)

        return mul, add

# Dense Connection
class Local_pred_share(nn.Module):
    def __init__(self, dim=16, number=3, type='ccc'):
        super(Local_pred_share, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.ReLU()

        self.block1 = CBlock_ln(dim, drop_path=0.02)
        self.block2 = CBlock_ln(dim, drop_path=0.04)
        self.block3 = CBlock_ln(dim, drop_path=0.06)
        self.block4 = CBlock_ln(dim, drop_path=0.08)

        # self.swin_add = SwinTransformerBlock(dim)
        # self.swin_mul = SwinTransformerBlock(dim)

        self.mul_head = nn.Sequential(nn.Conv2d(dim*4, dim, 1), SwinTransformerBlock(dim, drop_path=0.1))
        self.add_head = nn.Sequential(nn.Conv2d(dim*4, dim, 1), SwinTransformerBlock(dim, drop_path=0.1))
        # self.mul_head = nn.Sequential(nn.Conv2d(dim*4, dim, 1), CBlock_ln(dim, drop_path=0.1))
        # self.add_head = nn.Sequential(nn.Conv2d(dim*4, dim, 1), CBlock_ln(dim, drop_path=0.1))
        # self.add_end = nn.Sequential(nn.Conv2d(22, 3, 3, 1, 1))
        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        #self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, img):
        fea = self.relu(self.conv1(img)) 
        fea1 = self.block1(fea)
        fea2 = self.block2(fea1)
        fea3 = self.block3(fea2)
        fea4 = self.block4(fea3)
        fea_total = torch.cat([fea1, fea2, fea3, fea4], 1)

        mul = self.mul_end(self.mul_head(fea_total))
        add = self.add_end(self.add_head(fea_total))

        return mul, add

class Local_pred_share_v1(nn.Module):
    def __init__(self, dim=16, number=3, type='ccc'):
        super(Local_pred_share_v1, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.ReLU()

        self.block1 = CBlock_ln(dim, drop_path=0.025)
        self.block2 = CBlock_ln(dim, drop_path=0.05)
        self.block3 = CBlock_ln(dim, drop_path=0.075)
        #self.block4 = CBlock_ln(dim, drop_path=0.08)
        self.block4 = nn.Conv2d(dim*3, dim, 1)

        #self.mul_head = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.PReLU(), SwinTransformerBlock(dim, drop_path=0.1))
        #self.add_head = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.PReLU(), SwinTransformerBlock(dim, drop_path=0.1))
        self.mul_head = CBlock_ln(dim, drop_path=0.1)
        self.add_head = CBlock_ln(dim, drop_path=0.1)
        # self.add_end = nn.Sequential(nn.Conv2d(22, 3, 3, 1, 1))
        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1))
        #self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, img):
        fea = self.relu(self.conv1(img)) 
        fea1 = self.block1(fea)
        fea2 = self.block2(fea1)
        fea3 = self.block3(fea2)
        #fea4 = self.block4(fea3)
        fea_total = torch.cat([fea1, fea2, fea3], 1)
        fea_total = self.block4(fea_total) + fea
        
        mul = self.mul_end(self.mul_head(fea_total))
        add = self.add_end(self.add_head(fea_total))

        return mul, add

class Global_pred(nn.Module):
    def __init__(self, number=2, dim=16):
        super(Global_pred, self).__init__()
        #self.gamma_base = 1.0  # basic gamma value
        self.color_base = torch.eye(3)  # basic color matrix
        
        # main blocks
        self.conv_large = nn.Conv2d(3, dim, 7, padding=3, groups=1, stride=4)
        block = CBlock_ln(dim)
        blocks1, blocks2 = [block for _ in range(number)], [block for _ in range(number)]
        #self.gamma_blocks = nn.Sequential(*blocks1)
        self.color_blocks = nn.Sequential(*blocks2)
        #self.gamma_head = nn.Linear(dim, 1)
        self.color_head = nn.Linear(dim, 9)

    def forward(self, img):
        img = self.conv_large(img)
        color = self.color_blocks(img)
        #gamma = self.gamma_blocks(img)
        #B, C, H, W = color.shape
        #gamma = F.adaptive_avg_pool2d(gamma, 1).squeeze(-1).squeeze(-1)
        #gamma = self.gamma_head(gamma) + self.gamma_base
        color = F.adaptive_avg_pool2d(color, 1).squeeze(-1).squeeze(-1)
        color = self.color_head(color).view(-1,3,3)
        
        device = color.device
        color = 0.5*color+torch.stack([self.color_base.to(torch.device(device)) for i in range(color.shape[0])],dim=0)
        
        return color

class Dynamic_SID(nn.Module):
    def __init__(self):
        super(Dynamic_SID, self).__init__()
        #self.local_net = Local_pred_new()
        self.local_net = Local_pred_share_v1()
        self.global_net = Global_pred()

        # init
        #self.apply(self._init_weights)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

    def forward(self, img_low, with_global = False):
        mul, add = self.local_net(img_low)

        # function [a(x)*x+b(x)]**gamma*color
        img_high = (img_low.mul(mul)).add(add)

        if not with_global:
            return mul, add, img_high

        else:
            color, gamma = self.global_net(img_low)
            b = img_high.shape[0]
            img_high = img_high.permute(0,2,3,1)  # (B,C,H,W) -- (B,H,W,C)
            img_high = torch.stack([self.apply_color(img_high[i,:,:,:], color[i,:,:])**gamma[i,:] for i in range(b)], dim=0)
            img_high = img_high.permute(0,3,1,2)  # (B,H,W,C) -- (B,C,H,W)

            return mul, add, img_high




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    net = Local_pred_new().cuda()
    #img = torch.Tensor(8, 3, 400, 600)
    #mul, add = net(img)
    #print(img_high.shape)
    summary(net, input_size=(3, 400, 600))
    # mul, add = local_net(img)
    # print(mul.shape, add.shape)


