# Low-Light Object Detection (EXDark dataset)

**Step 1: Dataset Download**

(1). Download **EXDark** (include images enhancement by MBLLEN, Zero-DCE, KIND) in VOC format from [google drive](https://drive.google.com/file/d/1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC/view?usp=sharing) or [baiduyun](https://pan.baidu.com/s/1m4BMVqClhMks4S0xulkCcA), passwd:1234. For linux system download (google drive), directly run: 

```
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC" -O EXDark.tar.gz && rm -rf /tmp/cookies.txt
```

(2). Then unzip:

```
$ tar -zxvf EXDark.tar.gz
```

We have already split the EXDark dataset with train (80%) and test set (20%).

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

(3). Then change [line1](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/a0e4de1029eab1e6030f11cebbb7aaec2a64360b/IAT_high/IAT_mmdetection/configs/_base_/datasets/exdark_detr.py#L3) and [line2](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/a0e4de1029eab1e6030f11cebbb7aaec2a64360b/IAT_high/IAT_mmdetection/configs/_base_/datasets/exdark_yolo.py#L2) to your own data path.


**Step 2: Enviroment Setting**

Download mmcv 1.3.8~1.4.0, and download adapte to your own cuda version and torch version:
```
$ pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
```
then set up mmdet (2.15.1):
```
$ pip install -r requirements/build.txt
$ pip install -v -e .
```

**Step 3: Testing with pretrain model**

DETR pretrain models and training logs ([Baiduyun](https://pan.baidu.com/s/1CMAdhZ_9KvPnLfO7lyyaUA), passwd: 5hvr) or [Google Drive]().

YOLO-V3 pretrain models and training logs ([Baiduyun](https://pan.baidu.com/s/1tPXOBNC-6XElwvoIMPGPXQ), passwd: m6u4) or [Google Drive]().

Example of evaluation IAT-DETR-model:

```
$ python tools/test.py configs/detr/detr_ours_LOL.py DETR/detr_IAT_LOLpre.pth --eval mAP

$ python tools/test.py configs/detr/detr_ours_MIT5k.py DETR/detr_IAT_MIT5Kpre.pth --eval mAP
```

Example of evaluation IAT-YOLO-V3-model:

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

