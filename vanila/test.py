import sys
import cv2
import numpy as np
import torch
from networks.VDN import VDN
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import img_as_float, img_as_ubyte
from pathlib import Path
from utils import peaks, sincos_kernel, generate_gauss_kernel_mix, load_state_dict_cpu
import time
from dataset import SimulateH5
dataset = SimulateH5('/home/eunu/nas/DLA/gaussian.h5', 128, 3)

dataloader = torch.utils.data.DataLoader(dataset= dataset, batch_size=4)

for i, data in enumerate(dataloader):
    _, _, _, noise2 = data 
    print(len(data))
    break