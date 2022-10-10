_base_ = './deeplabv3plus_r50-d8_1024x1024_80k_ACDC_night.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
