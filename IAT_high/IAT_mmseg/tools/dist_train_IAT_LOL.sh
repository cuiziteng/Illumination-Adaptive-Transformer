#!/usr/bin/env bash


CONFIG_IAT=configs/deeplabv3plus/deeplabv3plus_r50-d8_1024x1024_80k_ACDC_night_with_IAT_LOL.py
LOAD_FROM=pre_train_models/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth
WORK_DIR=work_dir/IAT_LOL/
GPUS=2
PORT=29510

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=6,7  python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG_IAT --load-from $LOAD_FROM --work-dir $WORK_DIR  --launcher pytorch ${@:3} 
