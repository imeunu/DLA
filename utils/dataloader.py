import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from vanila.data_tools import random_augmentation

class ImageDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.sigma_min = 0
        self.sigma_max = 75
        self.pch_size = args.pch_size
        self.win = 2*args.radius+1
        self.sigma_spatial = args.radius

        if args.noise_type == 'gaussian':
            with h5py.File('/home/eunu/gaussian.h5', 'r') as f:
                self.imgs = np.array([np.array(f[key]) for key in f.keys()])

        else: # args.noise_type == 'real'
            with h5py.File('/home/eunu/real_gt.h5', 'r') as f1:
                self.imgs = np.array([np.array(f1[key]) for key in f1.keys()])
            with h5py.File('/home/eunu/real_.noisy.h5', 'r') as f2:
                self.noisy = np.array([np.array(f2[key]) for key in f2.keys()])

    def __getitem__(self, idx):

        # if self.args.strategy == 'std':
        #     noisy = torch.Tensor(self.add_noise(clean))
        #     return (noisy, clean)

        # elif self.args.strategy == 'n2n':
        #     noisy = torch.Tensor(self.add_noise(clean))
        #     target = torch.Tensor(self.add_noise(clean))
        #     return (noisy, target)

        # elif self.args.strategy == 'n2v':
        #     return ()

        clean = torch.Tensor(self.imgs[idx])
        if self.args.noise_type == 'real':
            noisy = torch.Tensor(self.noisy[idx])
        else:
            sigma_map = self.generate_sigma()
            noise = np.random.randn(clean.shape) * sigma_map
            noisy = np.clip(clean.astype(np.float64) + noise, 0, 255)
        clean, noisy = crop_patch(clean, noisy)
        clean, noisy = random_augmentation(clean, noisy, sigma_map)

        sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial)
        sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2,0,1)))
        sigma2_map_gt = np.tile(np.square(sigma_map), (1,1,clean.shape[2]))
        sigma2_map_gt = np.where(sigma2_map_gt<1e-10, 1e-10, sigma2_map_gt)
        sigma2_map_gt =  torch.from_numpy(sigma2_map_gt.transpose((2,0,1)))
        im_gt = torch.from_numpy(im_gt.transpose((2,0,1)))
        im_noisy = torch.from_numpy(im_noisy.transpose(2,0,1))
        
        return im_noisy, im_gt, sigma2_map_est, sigma2_map_gt

    def __len__(self):
        return len(self.imgs)
    
    def add_noise(self, img, type='gaussian'):
        if type == 'gaussian':
            img_size = np.shape(img)
            noise = np.random.normal(0, self.args.sigma, img_size)
            noisy = img + noise
        return np.clip(noisy, 0, 255)

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
    
    def crop_patch(self, clean, noisy): 
        H = clean.shape[0]
        W = clean.shape[1]
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W-self.pch_size)
        clean_pch = clean[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size]
        noisy_pch = noisy[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size]
        return clean_pch, noisy_pch