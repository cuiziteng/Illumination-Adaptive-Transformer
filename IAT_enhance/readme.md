# IAT in Low-Level Vision

## Low-Light Enhancement (LOL dataset)

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

2. Evaluation pretrain model on LOL dataset
```
python evaluation_lol.py --img_val_path Your_Path/Test/Low 
```
The SSIM and PSNR value should be **0.823** and **23.499**

3. Training your model on LOL dataset
```
python train_lol.py --gpu_id 0 --img_path Your_Path/Train/Low --img_val_path Your_Path/Test/Low 
```
