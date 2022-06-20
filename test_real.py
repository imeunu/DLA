import torch 
from torch.utils.data import DataLoader, Dataset
import h5py as h5
import numpy as np
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio
from vanila.networks.VDN import VDN
import cv2 
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

'''im_noisy, im_gt = dataset.__getitem__(19)

im_noisy.cpu().numpy()
im_gt.cpu().numpy()
im_noisy = np.transpose(im_noisy, (1,2,0))
im_gt = np.transpose(im_gt, (1,2,0))

im_noisy = img_as_ubyte(im_noisy)
im_gt = img_as_ubyte(im_gt)

cv2.imwrite('./results_real/noise.jpg', im_noisy)
cv2.imwrite('./results_real/gt.jpg', im_gt)'''

data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle= False)

# set devive 
device = torch.device('cuda:2')

# Load model 
print('Loading the Model')
checkpoint = torch.load('/home/junsung/DLA/N2N/model_N2N/model_state_150.pth', map_location=device)

checkpoint = torch.load('/home/eunu/nas/DLA/model_state_150.pth', map_location=device)

#----------for data parallel error---------# 
from collections import OrderedDict

new_state_dict =OrderedDict()
for k, v in checkpoint.items():
    name = k[7:]
    new_state_dict[name] = v
#-----------------------------------------#
net = VDN(3)
net.load_state_dict(new_state_dict)
net.to(device)
net.eval()

total_psnr = 0
for data in data_loader:
    im_noisy, im_gt = data 
    im_noisy = im_noisy.to(device)

    with torch.set_grad_enabled(False):
        phi_Z = net(im_noisy, 'test')
        err = phi_Z.cpu().numpy()
    
    im_noisy = im_noisy.cpu().numpy()

    im_denoise = im_noisy - err[:, :3,]
    im_denoise = np.transpose(im_denoise.squeeze(), (1,2,0))
    im_denoise = img_as_ubyte(im_denoise.clip(0,1))
    
    im_noisy = np.transpose(im_noisy.squeeze(), (1,2,0))
    im_noisy = img_as_ubyte(im_noisy.clip(0,1))
    im_gt = np.transpose(im_gt.squeeze(), (1,2,0))
    im_gt = img_as_ubyte(im_gt)
    psnr_val = peak_signal_noise_ratio(im_gt, im_denoise, data_range=255)

    total_psnr += psnr_val

print('psnr : ', total_psnr / dataset.__len__())