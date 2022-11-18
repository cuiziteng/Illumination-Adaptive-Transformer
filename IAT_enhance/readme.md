# IAT in Low-Level Vision

## I. Low-Light Enhancement (LOL-V1 dataset, 485 training image, 15 testing image)

1. Download the dataset from the [here](https://daooshee.github.io/BMVC2018website/). The dataset should contains 485 training image and 15 testing image, and should format like:

```
Your_Path
  -- our485
      -- high
      -- low
  -- eval15
      -- high
      -- low
```

2. Evaluation pretrain model on LOL-V1 dataset
```
python evaluation_lol_v1.py --img_val_path Your_Path/eval15/low/
```

Results:
|    | SSIM  | PSNR | enhancement images |
| -- | -- | -- | -- |
|  results  | **0.809**  |  **23.38** | [Baidu Cloud](https://pan.baidu.com/s/1M3H5coIOwfzYdTbZCkM42g) (passwd: 5pj2), [Google Drive](https://drive.google.com/drive/folders/1fgDUEbdiRkLbORZt4LMTX5rFB_erexOc?usp=sharing)|

3. Training your model on LOL-V1 dataset (get our closely result).

Step 1: crop the LOL-V1 dataset to 256 $\times$ 256 patches:
```
python LOL_patch.py --src_dir Your_Path/our485 --tar_dir Your_Path/our485_patch
```

Step 2: train on LOL-V1 patch images:
```
python train_lol_v1_patch.py --img_path Your_Path/our485_patch/low/ --img_val_path Your_Path/eval15/low/
```

Step 3: tuned the pre-train model (in Step 2) on LOL-V1 patches on the full resolution LOL-V1 image:
```
python train_lol_v1_whole.py --img_path Your_Path/our485/low/ --img_val_path Your_Path/eval15/low/ --pretrain_dir workdirs/snapshots_folder_lol_v1_patch/best_Epoch.pth
```

<br/>

## II. Low-Light Enhancement (LOL-V2-real dataset, 589 training image, 100 testing image)

1. Download the dataset from [Baidu_Cloud](https://pan.baidu.com/s/1Md5r4Lup8NVQI2ixKTIlGQ)(passwd: m7f7) or [Google Drive](https://drive.google.com/file/d/17UiWwwLHHveHf7N2Ubknpk7FUsN06W6a/view?usp=sharing), the dataset should format like:

```
Your_Path
  -- Train
      -- Normal
      -- Low
  -- Test
      -- Normal
      -- Low
```

2. Evaluation pretrain model on LOL-V2-real dataset
```
python evaluation_lol_v2.py --img_val_path Your_Path/Test/Low/
```
Results:

|  | SSIM | PSNR | enhancement images |
| -- | -- | -- | -- |
| results | **0.824** | **23.50**  | [Baidu Cloud](https://pan.baidu.com/s/1XH8Bpo0UgrJEqz_gOefiQA)(passwd: 6u3m), [Google Drive](https://drive.google.com/drive/folders/1rxBGGLIguNP0r_Of4dxQ1VAZRnGYJZGu?usp=sharing)|

3. Training your model on LOL-V2-real dataset (single GPU), for LOL-V2-real, you don't need create patch and directly train is OK.
```
python train_lol_v2.py --gpu_id 0 --img_path Your_Path/Train/Low --img_val_path Your_Path/Test/Low/ 
```

<br/>

## III. Exposure Correction

1. Download the dataset from [Training](https://ln2.sync.com/dl/141f68cf0/mrt3jtm9-ywbdrvtw-avba76t4-w6fw8fzj), [Validation](https://ln2.sync.com/dl/49a6738c0/3m3imxpe-w6eqiczn-vripaqcf-jpswtcfr), [Testing](https://ln2.sync.com/dl/098a6c5e0/cienw23w-usca2rgh-u5fxikex-q7vydzkp), then the dataset should format like:

```
Your_Path
  -- train
      -- GT_IMAGES
      -- INPUT_IMAGES
  -- validation
      -- GT_IMAGES
      -- INPUT_IMAGES
  -- test
      -- INPUT_IMAGES
      -- expert_a_testing_set
      -- expert_b_testing_set
      -- expert_c_testing_set
      -- expert_d_testing_set
      -- expert_e_testing_set
```

2. Evaluation pretrain model on Exposure dataset
```
python evaluation_exposure.py --gpu_id 0 --img_val_path Your_Path/test/INPUT_IMAGES/ --expert a/b/c/d/e (choose 1)
```

The results should be:

| **Expert** | a | b | c | d | e |
| -- | -- | -- | -- | -- | -- |
| PSNR | 19.62 | 21.24 | 21.27 | 19.74 | 19.51 |
| SSIM | 0.80 | 0.83 | 0.84 | 0.82 | 0.82 | 

3. Training your model on Exposure dataset (multi-GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 python -m torch.distributed.launch --nproc_per_node=4 train_exposure.py --img_path Your_Path/train/INPUT_IMAGES --img_val_path Your_Path/validation/INPUT_IMAGES
```

<br/>

## Others:

1. To use the model for a exposure correction demo show, direct run:
```
python img_demo.py --file_name demo_imgs/exposure_demo.JPG --task exposure
```

or for a image enhancement demo show, direct run:
```
python img_demo.py --file_name demo_imgs/low_demo.jpg --task enhance
```


2. To check how many parameters in IAT model, direct run:
```
python model/IAT_main.py
```

Dataset Citation:

```
@inproceedings{LOL_dataset,
  title={Deep Retinex Decomposition for Low-Light Enhancement},
  author={Chen Wei and Wenjing Wang and Wenhan Yang and Jiaying Liu},
  booktitle={British Machine Vision Conference},
  year={2018},
}

@InProceedings{Exposure_2021_CVPR,
    author    = {Afifi, Mahmoud and Derpanis, Konstantinos G. and Ommer, Bjorn and Brown, Michael S.},
    title     = {Learning Multi-Scale Photo Exposure Correction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition },
    year      = {2021},
}
```
