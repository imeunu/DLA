import sys
sys.path.append('./')
import os 
import cv2
import numpy as np
import torch
from vanila.networks.VDN import VDN
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float, img_as_ubyte
from vanila.utils import peaks, sincos_kernel, generate_gauss_kernel_mix, load_state_dict_cpu
from matplotlib import pyplot as plt
import time

use_gpu = True
case = 3
C = 3
dep_U = 4

# device 
device = torch.device('cuda:2')

# load the pretrained model
print('Loading the Model')

#checkpoint = torch.load('/home/junsung/DLA/N2N/model_N2N/model_state_150.pth', map_location=device)
checkpoint = torch.load('/home/eunu/nas/DLA/model_state_150.pth', map_location=device)

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

im_gt = img_as_float(cv2.imread('./results_niid/gt.jpg')[:, :, ::-1])
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

#im_noisy = img_as_float(cv2.imread('./results_real/noise.jpg')[:, :, ::-1])
#im_noisy = np.clip(im_noisy, 0,1).astype(np.float32)
im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])

'''noise = np.random.normal(0,25, im_gt.shape) / 255.0
im_noisy = np.clip(im_gt + noise, 0, 1).astype(np.float32)

im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])
'''
if use_gpu:
    im_noisy = im_noisy.to(device)

with torch.autograd.set_grad_enabled(False):
    tic = time.perf_counter()
    phi_Z = net(im_noisy, 'test')
    toc = time.perf_counter()
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

#cv2.imwrite('./results_niid/noise.jpg', cv2.cvtColor(im_noisy, cv2.COLOR_BGR2RGB))
cv2.imwrite('./results_niid/VDN.jpg', cv2.cvtColor(im_denoise, cv2.COLOR_BGR2RGB))