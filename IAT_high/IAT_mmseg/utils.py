import os

import cv2
from PIL import Image
import time

def generate_img_info_ACDC(ACDC_night_data_dir: str, save_dir: str):
    for root, dirs, _ in os.walk(ACDC_night_data_dir):
        # root='D:\\iLeaning\\CV\\dataset\\ACDC\\gt\\night'
        # dirs=['train', 'val']
        for sub_dir in dirs:
            # sub_dir='train'
            # 文件列表
            addr_list = os.listdir(os.path.join(root, sub_dir))
            for addr in addr_list:
                # addr='GOPR0351'
                image_list = os.listdir(os.path.join(root, sub_dir, addr))
                # 取出 _gt_labelTrainIds.png 结尾的文件
                image_list = list(filter(lambda x: x.endswith('_gt_labelTrainIds.png'), image_list))
                for image in image_list:
                    time_start = time.time()
                    # image='GOPR0351_frame_000159_gt_labelTrainIds.png'
                    image_dir = os.path.join(root, sub_dir, addr, image)
                    label_data = edit_gt_label(image_dir)
                    _save_dir = os.path.join(save_dir, sub_dir, addr, image)
                    label_data_new = Image.fromarray(label_data)
                    label_data_new.save(_save_dir)
                    time_end = time.time()
                    print('time cost', time_end - time_start, 's')

def edit_gt_label(train_root):
    label_data = cv2.imread(train_root)
    H, W, C = label_data.shape
    for i in range(H):
        for j in range(W):
            if label_data[i, j, 0] == 12 or label_data[i, j, 0] == 14 \
                    or label_data[i, j, 0] == 15 or label_data[i, j, 0] == 17:
                label_data[i, j, 0] = 255
                label_data[i, j, 1] = 255
                label_data[i, j, 2] = 255
    return label_data


if __name__ == '__main__':
    ACDC_night_data_dir = '/home/ssh685/CV_PROJECT/mmsegmentation-master/data/ACDC_night_1/gt'
    save_dir = '/home/ssh685/CV_PROJECT/mmsegmentation-master/data/new_gt_labelTrainIds'
    generate_img_info_ACDC(ACDC_night_data_dir=ACDC_night_data_dir, save_dir=save_dir)

