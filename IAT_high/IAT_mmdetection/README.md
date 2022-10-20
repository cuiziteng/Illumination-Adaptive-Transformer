# Low-Light Object Detection (EXDark dataset)

**Step 1: Dataset Download**

(1). Download **EXDark** (include images enhancement by MBLLEN, Zero-DCE, KIND) in VOC format from [google drive](https://drive.google.com/file/d/1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC/view?usp=sharing) or [baiduyun](https://pan.baidu.com/s/1m4BMVqClhMks4S0xulkCcA), passwd:1234. For linux system download, directly run: 

```
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC" -O EXDark.tar.gz && rm -rf /tmp/cookies.txt
```

(2). Then unzip:

```
$ tar -zxvf EXDark.tar.gz
```

We have already split the EXDark dataset with train set (80%) and test set (20%), see paper [MAET (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.pdf) for more details.

The EXDark dataset format should be look like:

```
EXDark
│      
│
└───JPEGImages
│   │───IMGS (original low light)
│   │───IMGS_Kind (imgs enhancement by [Kind, mm 2019])
│   │───IMGS_ZeroDCE (imgs enhancement by [ZeroDCE, cvpr 2020])
│   │───IMGS_MEBBLN (imgs enhancement by [MEBBLN, bmvc 2018])
│───Annotations   
│───main
│───label
```

(3). Then change [line1](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/a0e4de1029eab1e6030f11cebbb7aaec2a64360b/IAT_high/IAT_mmdetection/configs/_base_/datasets/exdark_detr.py#L3) (IAT_high/IAT_mmdetection/configs/_base_/datasets/exdark_detr.py) and [line2](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/a0e4de1029eab1e6030f11cebbb7aaec2a64360b/IAT_high/IAT_mmdetection/configs/_base_/datasets/exdark_yolo.py#L2) (IAT_high/IAT_mmdetection/configs/_base_/datasets/exdark_yolo.py) to your own data path.


**Step 2: Enviroment Setting**

Download mmcv 1.3.8~1.4.0, and download adapte to your own cuda version and torch version:
```
$ pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
```
then set up mmdet (2.15.1):
```
$ pip install opencv-python scipy
$ pip install -r requirements/build.txt
$ pip install -v -e .
```

**Step 3: Testing with pretrain model**

DETR pretrain models and training logs ([Baiduyun](https://pan.baidu.com/s/1CMAdhZ_9KvPnLfO7lyyaUA), passwd: 5hvr) or [Google Drive](https://drive.google.com/drive/folders/1ohpMnVsgzGi_k2GTgvJwZ3zlHSX1r5LX?usp=sharing).

YOLO-V3 pretrain models and training logs ([Baiduyun](https://pan.baidu.com/s/1tPXOBNC-6XElwvoIMPGPXQ), passwd: m6u4) or [Google Drive](https://drive.google.com/drive/folders/1n0Pi9kgeMF-nKXpS4-DLEoTwX_T9Lasz?usp=sharing).

Example of evaluation IAT-DETR-model (trained with LOL dataset/ MIT5K dataset weights):

```
$ python tools/test.py configs/detr/detr_ours_LOL.py DETR/detr_IAT_LOLpre.pth --eval mAP

$ python tools/test.py configs/detr/detr_ours_MIT5k.py DETR/detr_IAT_MIT5Kpre.pth --eval mAP
```

Example of evaluation IAT-YOLO-V3-model (trained with LOL dataset/ MIT5K dataset weights):

```
$ python tools/test.py configs/yolo/yolov3_IAT_lol.py YOLO_V3/YOLOV3_IAT_LOLpre.pth --eval mAP

$ python tools/test.py configs/yolo/yolov3_IAT_mit5k.py YOLO_V3/YOLOV3_IAT_MIT5Kpre.pth --eval mAP
```

For more baseline models, please see this repo: [MAET (ICCV 2021)](https://github.com/cuiziteng/ICCV_MAET).

**Step 4: Training your own model**

By default, the DETR is trained on 2 GPUs with per GPU batch size 2 (2 x 2): 

```
$ CUDA_VISIBLE_DEVICES=0,1 PORT=29501 bash tools/dist_train.sh configs/detr/detr_ours_LOL.py 2

or

$ CUDA_VISIBLE_DEVICES=0,1 PORT=29501 bash tools/dist_train.sh configs/detr/detr_ours_MIT5k.py 2
```

and YOLOV3 is trained on 1 GPUs with per GPU batch size 8 (1 x 8):

```
$ python tools/train.py configs/yolo/yolov3_IAT_lol.py --gpu-ids 0

or

$ python tools/train.py configs/yolo/yolov3_IAT_mit5k.py --gpu-ids 0
```


**Others:**

*Baselines on EXDark dataset (renew) on YOLO-V3 object detector:*

| class | Bicycle | Boat | Bottle | Bus | Car | Cat | Chair | Cup | Dog | Motorbike | People | Table | Total |
|  ---- | ----    | ---- | ----   | ----| ----| ----| ----  | ----| ----| ----      | ----   |  ---- |  ---- |
| Baseline |79.8 | 75.3 | 78.1 | 92.3 | 83.0 | 68.0 | 69.0 | 79.0 | 78.0 | 77.3 | 81.5 | 55.5 | 76.4 |
| KIND (MM 2019) | 80.1 | 77.7 | 77.2 | 93.8 | 83.9 | 66.9 | 68.7 | 77.4 | 79.3 | 75.3 | 80.9 | 53.8 | 76.3 |
| MBLLEN (BMVC 2018) | 82.0 | 77.3 | 76.5 | 91.3 | 84.0 | 67.6 | 69.1 | 77.6 | 80.4 | 75.6 | 81.9 | 58.6 | 76.8 |
| Zero-DCE (CVPR 2020) | 84.1 | 77.6 | 78.3 | 93.1 | 83.7 | 70.3 | 69.8 | 77.6 | 77.4 | 76.3 | 81.0 | 53.6 | 76.9 |
| [MAET (ICCV 2021)](https://github.com/cuiziteng/ICCV_MAET) | 83.1| 78.5| 75.6| 92.9| 83.1| 73.4| 71.3| 79.0| 79.8| 77.2| 81.1| 57.0| 77.7|
| IAT-YOLOV3 (ours) | 79.8 | 76.9 | 78.6 | 92.5 | 83.8 | 73.6 | 72.4 | 78.6 | 79.0 | 79.0 | 81.1 | 57.7 | **77.8** |

Dataset Citation:

```
@article{EXDark,
  title={Getting to know low-light images with the exclusively dark dataset},
  author={Loh, Yuen Peng and Chan, Chee Seng},
  journal={Computer Vision and Image Understanding},
  year={2019},
}
```

Code Usage Citation:

```
@InProceedings{Cui_2021_ICCV,
    author    = {Cui, Ziteng and Qi, Guo-Jun and Gu, Lin and You, Shaodi and Zhang, Zenghui and Harada, Tatsuya},
    title     = {Multitask AET With Orthogonal Tangent Regularity for Dark Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2553-2562}
}
```

