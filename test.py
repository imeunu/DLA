import argparse
import sys
sys.path.append('./')
import os 
import cv2
import numpy as np
import torch
import torch.nn as nn
from vanila.networks.VDN import VDN
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float, img_as_ubyte
from vanila.utils import peaks, sincos_kernel, generate_gauss_kernel_mix, load_state_dict_cpu
from matplotlib import pyplot as plt
import time


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type = bool, default=True)
    parser.add_argument('--checkpoint', type=str, default='/home/eunu/nas/DLA/model_state_150.pth')
    
    return parser.parse_args()

args = get_arguments()

use_gpu = args.use_gpu
case = 3
C = 3
dep_U = 4

checkpoints = {
    'vdn' : '/home/eunu/nas/DLA/model_state_150.pth',
    'N2N' : '/home/eunu/nas/DLA/n2n_sigma25_model_state_150.pth',
    'N2N_1' : '/home/eunu/nas/DLA/n2n_sigma25_model_state_150_1.pth',
    'unet' : '/home/eunu/nas/DLA/unet_sigma_25/unet_state_200.pth',
    'ft_vdn' : '/home/junsung/DLA/vanila/model/test/model_state_150.pth',
    'ft_N2N' : '/home/junsung/DLA/vanila/model/N2N/model_state_150.pth',
    'ft_ch_sigma_N2N' : '/home/junsung/DLA/vanila/model/N2N_ch_sigma/model_state_150.pth' 
}


# device 
device = torch.device('cuda:3')

model_name = 'N2N'
# load the pretrained model
print('Loading the Model')
checkpoint = torch.load(checkpoints[model_name], map_location=device)
#----------for data parallel error---------# 
from collections import OrderedDict

new_state_dict =OrderedDict()
for k, v in checkpoint.items():
    name = k[7:]
    new_state_dict[name] = v
#-----------------------------------------#
net = VDN(C, dep_U=dep_U, wf=64)
if use_gpu:
    net.load_state_dict(new_state_dict)
    net.to(device)
else:
    load_state_dict_cpu(net, checkpoint)
net.eval()

datasets = ['Set5', 'LIVE1', 'CBSD68']
print(model_name)
print('---------------------Synthetic results------------------------')
for case in range(1,4): 
    for dataset in datasets:
        psnr_total = 0
        im_lst = os.listdir('/home/eunu/nas/DLA/test_data/'+dataset)
        for im in im_lst:
            im_path = os.path.join('/home/eunu/nas/DLA/test_data', dataset, im)
            im_gt = img_as_float(cv2.imread(im_path)[:, :, ::-1])

            H, W, _ = im_gt.shape
            if H % 2**dep_U != 0:
                H -= H % 2**dep_U
            if W % 2**dep_U != 0:
                W -= W % 2**dep_U
            im_gt = im_gt[:H, :W, ]
            
            # Generate the sigma map
            if case == 1:
                # Test case 1
                sigma = peaks(256)
            elif case == 2:
                # Test case 2
                sigma = sincos_kernel()
            elif case == 3:
                # Test case 3
                sigma = generate_gauss_kernel_mix(256, 256)
            else:
                sys.exit('Please input the corrected test case: 1, 2 or 3')

            sigma = 5/255.0 + (sigma-sigma.min())/(sigma.max()-sigma.min()) * ((25)/255.0)
            sigma = cv2.resize(sigma, (W, H))
            noise = np.random.randn(H, W, C) * sigma[:, :, np.newaxis]
            im_noisy = np.clip(im_gt + noise, 0, 1).astype(np.float32)

            im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])

            if use_gpu:
                im_noisy = im_noisy.to(device)

            with torch.autograd.set_grad_enabled(False):
                phi_Z = net(im_noisy, 'test')
                err = phi_Z.cpu().numpy()
            if use_gpu:
                im_noisy = im_noisy.cpu().numpy()
            else:
                im_noisy = im_noisy.numpy()
            im_denoise = im_noisy - err[:, :C,]
            im_denoise = np.transpose(im_denoise.squeeze(), (1,2,0))
            im_denoise = img_as_ubyte(im_denoise.clip(0,1))

            im_noisy = np.transpose(im_noisy.squeeze(), (1,2,0))
            im_noisy = img_as_ubyte(im_noisy.clip(0,1))

            im_gt = img_as_ubyte(im_gt)
            psnr_val = peak_signal_noise_ratio(im_gt, im_denoise, data_range=255)
            
            psnr_total += psnr_val

        print('for case : {}, dataset: {}, psnr_avg : {}'.format(case, dataset, psnr_total / len(im_lst)) )

print('---------------------AWGN results------------------------')
sigmas = [15,25,50]
for sigma_level in sigmas:
    for dataset in datasets:
        psnr_total = 0
        im_lst = os.listdir('/home/eunu/nas/DLA/test_data/'+dataset)
        for im in im_lst:
            im_path = os.path.join('/home/eunu/nas/DLA/test_data', dataset, im)
            im_gt = img_as_float(cv2.imread(im_path)[:, :, ::-1])

            H, W, _ = im_gt.shape
            if H % 2**dep_U != 0:
                H -= H % 2**dep_U
            if W % 2**dep_U != 0:
                W -= W % 2**dep_U
            im_gt = im_gt[:H, :W, ]
            
            noise = np.random.normal(0,sigma_level,im_gt.shape)/255.0
            im_noisy = np.clip(im_gt + noise, 0, 1).astype(np.float32)

            im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])

            if use_gpu:
                im_noisy = im_noisy.to(device)

            with torch.autograd.set_grad_enabled(False):
                phi_Z = net(im_noisy, 'test')
                err = phi_Z.cpu().numpy()
            if use_gpu:
                im_noisy = im_noisy.cpu().numpy()
            else:
                im_noisy = im_noisy.numpy()
            im_denoise = im_noisy - err[:, :C,]
            im_denoise = np.transpose(im_denoise.squeeze(), (1,2,0))
            im_denoise = img_as_ubyte(im_denoise.clip(0,1))

            im_noisy = np.transpose(im_noisy.squeeze(), (1,2,0))
            im_noisy = img_as_ubyte(im_noisy.clip(0,1))

            im_gt = img_as_ubyte(im_gt)
            psnr_val = peak_signal_noise_ratio(im_gt, im_denoise, data_range=255)
            
            psnr_total += psnr_val

        print('for simga : {}, dataset: {}, psnr_avg : {}'.format(sigma_level, dataset, psnr_total / len(im_lst)) )