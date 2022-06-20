import torch 
from torch.utils.data import DataLoader, Dataset
import h5py as h5
import numpy as np
from skimage import img_as_float,img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float, img_as_ubyte
from vanila.utils import peaks, sincos_kernel, generate_gauss_kernel_mix, load_state_dict_cpu
from vanila.networks.VDN import VDN
from Unet import UNet
import cv2 
import sys 
import os 

class BenchmarkTest(Dataset):
    def __init__(self,h5_path):
        self.h5_path = h5_path        
        with h5.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(self.keys)

    
    def __getitem__(self, index):
        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2/2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
        im_gt = img_as_float32(im_gt)
        im_noisy = img_as_float32(im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt

    def __len__(self):
        return self.num_images

# dataset 
dataset = BenchmarkTest(h5_path='/home/eunu/nas/DLA/test_data/small_imgs_test.hdf5')
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle= False)

use_gpu = True
case = 3
C = 3
dep_U = 4

# device 
device = torch.device('cuda:2')

checkpoint = torch.load('/home/eunu/nas/DLA/model_state_70.pth', map_location=device)
#----------for data parallel error---------# 
from collections import OrderedDict

new_state_dict =OrderedDict()
for k, v in checkpoint.items():
    name = k[7:]
    new_state_dict[name] = v
#-----------------------------------------#

net = UNet()
if use_gpu:
    net.load_state_dict(new_state_dict)
    net.to(device)
net.eval()

im_noisy = img_as_float32(cv2.imread('./results_niid/gt.jpg')[:, :, ::-1])
H, W, _ = im_noisy.shape
if H % 2**dep_U != 0:
    H -= H % 2**dep_U
if W % 2**dep_U != 0:
    W -= W % 2**dep_U
im_gt = im_noisy[:H, :W, ]
im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])
im_noisy = im_noisy.to(device)
with torch.autograd.set_grad_enabled(False):
    denoise = net(im_noisy)
    im_denoise = denoise.cpu().numpy()
    

im_denoise=im_denoise.squeeze()
im_denoise = np.transpose(im_denoise, (1,2,0))
im_denoise = img_as_ubyte(im_denoise.clip(0,1))

cv2.imwrite('./results_niid/unet.jpg', cv2.cvtColor(im_denoise, cv2.COLOR_BGR2RGB))

sys.exit()

total_psnr = 0
for ii, data in enumerate(data_loader):
    im_noisy, im_gt = data 
    im_noisy = im_noisy.to(device)

    with torch.autograd.set_grad_enabled(False):
        denoise = net(im_noisy)
        im_denoise = denoise.cpu().numpy()
    
    im_noisy = im_noisy.cpu().numpy()

    im_denoise=im_denoise.squeeze()
    im_denoise = np.transpose(im_denoise, (1,2,0))
    im_denoise = img_as_ubyte(im_denoise.clip(0,1))
    im_noisy = np.transpose(im_noisy.squeeze(), (1,2,0))
    im_noisy = img_as_ubyte(im_noisy.clip(0,1))
    im_gt = np.transpose(im_gt.squeeze(), (1,2,0))
    im_gt = img_as_ubyte(im_gt)
    psnr_val = peak_signal_noise_ratio(im_gt, im_denoise, data_range=255)

    total_psnr += psnr_val

print('real psnr : ', total_psnr / ii+1)

datasets = ['Set5', 'LIVE1', 'CBSD68']

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

            noise = np.random.normal(0,50, im_gt.shape) / 255.0
            im_noisy = np.clip(im_gt + noise, 0, 1).astype(np.float32)
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
            #im_noisy = (im_gt + noise).astype(np.float32)

            im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])


            if use_gpu:
                im_noisy = im_noisy.to(device)

            with torch.autograd.set_grad_enabled(False):
                denoise = net(im_noisy)
                im_denoise = denoise.cpu().numpy()
            
            im_noisy = im_noisy.cpu().numpy()

            im_denoise=im_denoise.squeeze()
            im_denoise = np.transpose(im_denoise, (1,2,0))
            im_denoise = img_as_ubyte(im_denoise.clip(0,1))
            im_noisy = np.transpose(im_noisy.squeeze(), (1,2,0))
            im_noisy = img_as_ubyte(im_noisy.clip(0,1))
            im_gt = img_as_ubyte(im_gt)
            psnr_val = peak_signal_noise_ratio(im_gt, im_denoise, data_range=255)


            psnr_total += psnr_val

        print('for case : {}, dataset: {}, psnr_avg : {}'.format(case, dataset, psnr_total / len(im_lst)) )

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

        noise = np.random.normal(0,50, im_gt.shape) / 255.0
        im_noisy = np.clip(im_gt + noise, 0, 1).astype(np.float32)
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
        #im_noisy = (im_gt + noise).astype(np.float32)

        im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])


        if use_gpu:
            im_noisy = im_noisy.to(device)

        with torch.autograd.set_grad_enabled(False):
            denoise = net(im_noisy)
            im_denoise = denoise.cpu().numpy()
        
        im_noisy = im_noisy.cpu().numpy()

        im_denoise=im_denoise.squeeze()
        im_denoise = np.transpose(im_denoise, (1,2,0))
        im_denoise = img_as_ubyte(im_denoise.clip(0,1))
        im_noisy = np.transpose(im_noisy.squeeze(), (1,2,0))
        im_noisy = img_as_ubyte(im_noisy.clip(0,1))
        im_gt = img_as_ubyte(im_gt)
        psnr_val = peak_signal_noise_ratio(im_gt, im_denoise, data_range=255)


        psnr_total += psnr_val

    print('sigma25 with dataset: {}, psnr_avg : {}'.format(dataset, psnr_total / len(im_lst))) 