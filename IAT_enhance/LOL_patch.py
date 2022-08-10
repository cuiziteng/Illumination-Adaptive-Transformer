"""
Code change from MPRNet (https://github.com/swz30/MPRNet/blob/main/Denoising/generate_patches_SIDD.py), we use it for LOL dataset.
"""
from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='/data/unagi0/cui_data/light_dataset/LOL_v1/our485', type=str, help='Directory for full resolution images')    # For LOL-V1
parser.add_argument('--tar_dir', default='/data/unagi0/cui_data/light_dataset/LOL_v1/our485_patch',type=str, help='Directory for image patches')
parser.add_argument('--type', default='LOL-V1', type=str, help='The DataSet Type')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=10, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=6, type=int, help='Number of CPU Cores')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores

if args.type == 'LOL-V1':
    Low_patchDir = os.path.join(tar, 'low')
    High_patchDir = os.path.join(tar, 'high')

if args.type == 'LOL-V2':
    Low_patchDir = os.path.join(tar, 'Low')
    High_patchDir = os.path.join(tar, 'Normal')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(Low_patchDir)
os.makedirs(High_patchDir)

#get sorted folders
files = natsorted(glob(os.path.join(src, '*', '*.png')))

low_files, high_files = [], []

for file_ in files:
    filename = os.path.split(file_)[-2]
    print(filename)
    if args.type == 'LOL-V1':
        if 'low' in filename:
            low_files.append(file_)
        if 'high' in filename:
            high_files.append(file_)
    
    if args.type == 'LOL-V2':
        if 'Low' in filename:
            low_files.append(file_)
        if 'Normal' in filename:
            high_files.append(file_)
    


def save_files(i):
    noisy_file, clean_file = low_files[i], high_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)

    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]

        cv2.imwrite(os.path.join(Low_patchDir, '{}_{}.png'.format(i+1, j+1)), noisy_patch)
        cv2.imwrite(os.path.join(High_patchDir, '{}_{}.png'.format(i+1, j+1)), clean_patch)

Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(low_files))))

