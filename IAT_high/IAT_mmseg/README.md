# Low-Light Semantic Segmentation (ACDC dataset)

**Step 1: Dataset Download**

(1). Download **ACDC dataset** (include images enhancement by MBLLEN, Histograms Equalization, DeepLPF) in VOC format from [baiduyun](https://pan.baidu.com/s/1c59Qk62S9bw4VUG483f-Xw), passwd:5678. 

(2). Then unzip:

```
$ unzip data.zip
```

The ACDC dataset format should be look like:

```
data
│      
│───ACDC_night_new (original low light)
│   	│───rgb 
│   	└───new_gt_labelTrainIds
│───DeepLPF (imgs enhancement by [DeepLPF, cvpr 2020])
│   	│───rgb 
│   	└───new_gt_labelTrainIds
│───Histograms_Equalization (imgs enhancement by [Histograms Equalization])
│   	│───rgb 
│   	└───new_gt_labelTrainIds
└───MBLLEN (imgs enhancement by [MEBBLN, bmvc 2018])
   		│───rgb 
   		└───new_gt_labelTrainIds
```

(3). Then change

-  [line1](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/f358dd8c5fca0dea91b45d99d556ec9ce33bb052/IAT_high/IAT_mmseg/configs/_base_/datasets/ACDC_night.py#L3) ( `IAT_high/IAT_mmseg/configs/_base_/datasets/ACDC_night.py`) ,
-  [line2](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/f358dd8c5fca0dea91b45d99d556ec9ce33bb052/IAT_high/IAT_mmseg/configs/_base_/datasets/ACDC_night_with_DeepLPF.py#L3) ( `IAT_high/IAT_mmseg/configs/_base_/datasets/ACDC_night_with_DeepLPF.py`) ,
-  [line3](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/f358dd8c5fca0dea91b45d99d556ec9ce33bb052/IAT_high/IAT_mmseg/configs/_base_/datasets/ACDC_night_with_HE.py#L3) ( `IAT_high/IAT_mmseg/configs/_base_/datasets/ACDC_night_with_HE.py`) ,
-  [line4](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/f358dd8c5fca0dea91b45d99d556ec9ce33bb052/IAT_high/IAT_mmseg/configs/_base_/datasets/ACDC_night_with_MBLLEN.py#L3) ( `IAT_high/IAT_mmseg/configs/_base_/datasets/ACDC_night_with_MBLLEN.py`) ,

to your own data path.

Noted that, since there are few rider, truck, bus and motorcycle in ACDC night dataset, we remove these 4 classes.

**Step 2: Enviroment Setting**

Download mmcv 1.3.8~1.4.0, and download adapte to your own cuda version and torch version:

```
$ conda create -n mmcv_mmseg python=3.8
$ conda activate mmcv_mmseg
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
$ pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
```

then set up mmseg: 

```
$ pip install opencv-python scipy timm
$ cd IAT_mmsegmentation
$ pip install -v -e .
```

**Step 3: Testing with pretrain model**

Deeplabv3+ pretrain models and training logs ([Baiduyun](https://pan.baidu.com/s/1RT0OtnflLxi4FtAUeSnnBg), passwd: 5678).

Example of evaluation IAT-Deeplabv3+ model (trained with LOL dataset/ MIT5K dataset weights):

```
$ python tools/test.py configs/deeplabv3plus/deeplabv3plus_r50_IAT_MIT5K.py work_dir/IAT_MIT5K/iter_20000.pth --eval mIoU
or
$ python tools/test.py configs/deeplabv3plus/deeplabv3plus_r50_IAT_LOL.py work_dir/IAT_LOL/iter_20000.pth --eval mIoU
```

**Step 4: Training your own model**

By default, the Deeplabv3+ is trained on 2 GPUs with per GPU batch size 4 (4 x 2):

Then change `IAT_high/IAT_mmseg/configs/_base_/models/deeplabv3plus_r50-d8_wtih_IAT_LOL.py` and `IAT_high/IAT_mmseg/configs/_base_/models/deeplabv3plus_r50-d8_wtih_IAT_MIT5K.py` to your own IAT pretrain checkpoint path.

```
$ bash tools/dist_train_IAT_LOL.sh 
or
$ bash tools/dist_train_IAT_MIT5K.sh 
```

**Others:**

```
$ bash tools/dist_train_DeepLPF.sh 
$ bash tools/dist_train_HE.sh 
$ bash tools/dist_train_MBLLEN.sh 
```

**Baselines on ACDC dataset (renew) on Deeplabv3+ segmentor:**

| Method                  | mIoU      |
| ----------------------- | --------- |
| baseline                | 63.33     |
| Histograms Equalization | 61.90     |
| MBLLEN                  | 62.95     |
| DeepLPF                 | 61.88     |
| IAT (None)              | 61.54     |
| IAT (LOL)               | **63.77** |
| IAT (FiveK)             | 62.14     |

Dataset Citation:

```
@InProceedings{SDV21,
  author = {Sakaridis, Christos and Dai, Dengxin and Van Gool, Luc},
  title = {{ACDC}: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  year = 2021
}
```

Code Usage Citation:

```
@inproceedings{Lv2018MBLLEN,
 title={MBLLEN: Low-light Image/Video Enhancement Using CNNs},
 author={Feifan Lv, Feng Lu, Jianhua Wu, Chongsoon Lim},
 booktitle={British Machine Vision Conference (BMVC)},
 year={2018}
}

@InProceedings{Moran_2020_CVPR,
author = {Moran, Sean and Marza, Pierre and McDonagh, Steven and Parisot, Sarah and Slabaugh, Gregory},
title = {DeepLPF: Deep Local Parametric Filters for Image Enhancement},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
