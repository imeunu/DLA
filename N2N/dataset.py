import torch
import h5py as h5
import random
import cv2
import os
import numpy as np
import torch.utils.data as uData
from skimage import img_as_float32 as img_as_float
from data_tools import sigma_estimate, random_augmentation, gaussian_kernel

from glob import glob
from pathlib import Path

class SimulateH5(uData.Dataset):
    def __init__(self, h5_path, pch_size, radius, sigma):
        self.h5_path = h5_path
        self.pch_size = pch_size
        self.sigma_min = 0
        self.sigma_max = sigma

        self.win = 2*radius + 1
        self.sigma_spatial = radius

        with h5.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(self.keys)
        
        with h5.File('/home/eunu/nas/DLA/gaussian.h5', 'r') as f:
            self.imgs = np.array([np.array(f[key]) for key in f.keys()])

    def __getitem__(self, index):
        
        im_gt = img_as_float(self.crop_patch(self.imgs[index]))
        C = im_gt.shape[2]

        # generate sigmaMap 
        sigma_map = self.generate_sigma()

        # generate noise 
        noise = torch.randn(im_gt.shape).numpy() * sigma_map
        im_noisy = im_gt + noise.astype(np.float32)

        im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)

        # sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial)
        # sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2,0,1)))

        # ground truth sigmamap
        sigma2_map_gt = np.tile(np.square(sigma_map), (1,1,C))
        sigma2_map_gt = np.where(sigma2_map_gt<1e-10, 1e-10, sigma2_map_gt)
        sigma2_map_gt =  torch.from_numpy(sigma2_map_gt.transpose((2,0,1)))

        im_gt = torch.from_numpy(im_gt.transpose((2,0,1)))
        im_noisy = torch.from_numpy(im_noisy.transpose(2,0,1))
        
        return im_noisy, im_gt, sigma2_map_gt, #sigma2_map_est

    def __len__(self):
        return self.num_images

    def generate_sigma(self):
        center = [random.uniform(0, self.pch_size), random.uniform(0, self.pch_size)]
        scale = random.uniform(self.pch_size/4, self.pch_size/4*3)
        kernel = gaussian_kernel(self.pch_size, self.pch_size, center, scale)
        up = random.uniform(self.sigma_max/255.0, self.sigma_max/255.0)
        down = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min())  *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]

    def crop_patch(self, im): 
        H = im.shape[0]
        W = im.shape[1]
        if H < self.pch_size or W < self.pch_size:
            H = max(self.pch_size, H)
            W = max(self.pch_size, W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W-self.pch_size)
        pch = im[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size]
        return pch

class SimulateTest(uData.Dataset):
    def __init__(self, im_list, h5_path):
        super(SimulateTest, self).__init__()
        self.im_list = im_list
        self.h5_path = h5_path

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        im_gt = cv2.imread(self.im_list[index], 1)[:, :, ::-1]
        im_key = os.path.basename(self.im_list[index]).split('.')[0]
        C = im_gt.shape[2]

        with h5.File(self.h5_path, 'r') as h5_file:
            noise = np.array(h5_file[im_key][:,:,:C])
        H, W, _ = noise.shape
        im_gt = img_as_float(im_gt[:H, :W])
        im_noisy = im_gt + noise

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1))).type(torch.float32)
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1))).type(torch.float32)

        return im_noisy, im_gt

class SimulateH5N2N(uData.Dataset):
    def __init__(self, h5_path, pch_size, radius, sigma):
        self.h5_path = h5_path
        self.pch_size = pch_size
        self.sigma_min = 0
        self.sigma_max = sigma

        self.win = 2*radius + 1
        self.sigma_spatial = radius

        with h5.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(self.keys)
            self.imgs = np.array([np.array(h5_file[key]) for key in h5_file.keys()])

    def __getitem__(self, index):
        im_gt = img_as_float(self.crop_patch(self.imgs[index]))
        C = im_gt.shape[2]

        # generate sigmaMap 
        sigma_map1, sigma_map2 = self.generate_sigma(), self.generate_sigma()
        sigma_map = sigma_map1 + sigma_map2

        # generate noise 
        noise1 = torch.randn(im_gt.shape).numpy() * sigma_map1
        noise2 = torch.randn(im_gt.shape).numpy() * sigma_map2
        im_noisy = im_gt + noise1.astype(np.float32)
        im_gt = im_gt + noise2.astype(np.float32)

        im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)

        # sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial)
        # sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2,0,1)))
        
        '''       noise1 = torch.randn(im_gt.shape).numpy() * sigma_map1
        noise2 = torch.randn(im_gt.shape).numpy() * sigma_map2
        im_noisy = im_gt + noise.astype(np.float32)

        im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)

        sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial)
        sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2,0,1)))'''

        # ground truth sigmamap
        sigma2_map_gt = np.tile(np.square(sigma_map), (1,1,C))
        sigma2_map_gt = np.where(sigma2_map_gt<1e-10, 1e-10, sigma2_map_gt)
        sigma2_map_gt =  torch.from_numpy(sigma2_map_gt.transpose((2,0,1)))

        im_gt = torch.from_numpy(im_gt.transpose((2,0,1)))
        im_noisy = torch.from_numpy(im_noisy.transpose(2,0,1))
        
        return im_noisy, im_gt, sigma2_map_gt, #sigma2_map_est

    def __len__(self):
        return self.num_images

    def generate_sigma(self):
        center = [random.uniform(0, self.pch_size), random.uniform(0, self.pch_size)]
        scale = random.uniform(self.pch_size/4, self.pch_size/4*3)
        kernel = gaussian_kernel(self.pch_size, self.pch_size, center, scale)
        up = random.uniform(self.sigma_max/255.0, self.sigma_max/255.0)
        down = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min())  *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]

    def crop_patch(self, im): 
        H = im.shape[0]
        W = im.shape[1]
        if H < self.pch_size or W < self.pch_size:
            H = max(self.pch_size, H)
            W = max(self.pch_size, W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W-self.pch_size)
        pch = im[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size]
        return pch

class RealH5(uData.Dataset):
    def __init__(self, gt_h5_path, noise_h5_path):
        with h5.File(gt_h5_path, 'r') as h5_file:
            self.gt_keys = list(h5_file.keys())
            self.gt_imgs = np.array([np.array(h5_file[key]) for key in h5_file.keys()])
            self.gt_num_images = len(self.keys)


        with h5.File(noise_h5_path, 'r') as h5_file:
            self.noise_keys = list(h5_file.keys())
            self.noise_num_images = len(self.keys)
            self.noise_imgs = np.array([np.array(h5_file[key]) for key in h5_file.keys()])

    def __getitem__(self, index):
        im_gt = img_as_float(self.crop_patch(self.gt_imgs[index]))
        im_noisy = img_as_float(self.crop_patch(self.noise_imgs[index]))
        
        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        
        sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))
        eps2 = torch.tensor([self.eps2], dtype=torch.float32).reshape((1,1,1))

        sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2, 0, 1)))
        
        return im_noisy, im_gt, sigma2_map_est, eps2

    def crop_patch(self, im): 
        H = im.shape[0]
        W = im.shape[1]
        if H < self.pch_size or W < self.pch_size:
            H = max(self.pch_size, H)
            W = max(self.pch_size, W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W-self.pch_size)
        pch = im[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size]
        return pch    