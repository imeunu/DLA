import argparse
import os

import cv2
import h5py
import numpy as np


def walk(dir):
    for dirpath, _, files in os.walk(dir):
        for filename in files:
            yield dirpath, filename

def is_img(ext):
    if ext in ['.png', '.jpg', '.bmp']: return True
    else: return False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/Users/eunu/Desktop/code/vvdn/dataset/train')
    parser.add_argument('--h5_path', type=str, default='/Users/eunu/Desktop/code/vvdn/dataset/train/train.h5')

if __name__ == '__main__':
    args = get_args()
    h5 = h5py.File(f'{args.h5_path}','w')
    for folder, filename in walk(args.path):
        # with open(f'{folder}/{filename}','rb') as f:
        #     check_chars = f.read()[-2:]
        # if check_chars != b'\xffxd9':
        #     print(f'{folder}/{filename}')
        name, ext = os.path.splitext(filename)
        if not is_img(ext): continue
        img = cv2.imread(f'{folder}/{filename}').astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        categorh_name = folder.lstrip(args.path).lstrip('\\')
        h5.create_dataset('{category_name + filename}', data=img)
    h5.close()