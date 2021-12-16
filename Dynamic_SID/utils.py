import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

import math
import torchvision
from torchvision import models, transforms
from IQA_pytorch import SSIM, MS_SSIM

EPS = 1e-9
PI = 22.0 / 7.0
# calculate PSNR
class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))

def adjust_learning_rate(optimizer, epoch,  lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 20

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))


def visualization(img, img_path, iteration):

    # --- Visualization for Checking --- #
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    img = img.cpu().numpy()

    for i in range(img.shape[0]):
        # save name
        name = str(iteration) + '_' + str(i) + '.png'
        print(name)

        img_single = np.transpose(img[i, :, :, :], (1, 2, 0))
        # print(img_single)
        img_single = np.clip(img_single, 0, 1) * 255.0
        img_single = cv2.UMat(img_single).get()
        img_single = img_single / 255.0

        plt.imsave(os.path.join(img_path, name), img_single)

ssim = SSIM()
psnr = PSNR()

def validation(model, val_loader):

    ssim = SSIM()
    psnr = PSNR()
    ssim_list = []
    psnr_list = []
    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
            _, _, enhanced_img = model(low_img)
            # print(enhanced_img.shape)
        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        #ssim_value = ssim(enhanced_img, high_img).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        # print('The %d image SSIM value is %d:' %(i, ssim_value))
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print('The SSIM Value is:', SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)
    return SSIM_mean, PSNR_mean

# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)




