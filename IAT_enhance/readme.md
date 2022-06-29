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

## Exposure Correction

1. Download the dataset from [Training](https://ln2.sync.com/dl/141f68cf0/mrt3jtm9-ywbdrvtw-avba76t4-w6fw8fzj), [Validation](https://ln2.sync.com/dl/49a6738c0/3m3imxpe-w6eqiczn-vripaqcf-jpswtcfr), [Testing](https://ln2.sync.com/dl/098a6c5e0/cienw23w-usca2rgh-u5fxikex-q7vydzkp), then the dataset should format like:

```
Your Path
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
    -- ...
```
