import torch 
import torch.nn as nn
import numpy as np
from networks.VDN import VDN
from scipy.io import loadmat
import time
import matplotlib.pyplot as plt 
from utils import load_state_dict_cpu
from skimage import img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio

# channel
C = 3
dep_U = 4

''' For GPU 
# device 
device = torch.device('cuda:2')

checkpoint = torch.load('./model_real/model_state_1500', map_location='cuda:2')
net = VDN(C, wf=64,dep_S=5, dep_U = 4, slope = 0.2)

net = nn.DataParallel(net).to(device)
net.load_state_dict(checkpoint)'''

checkpoint = torch.load('./model_real/model_state_1500', map_location='cpu')
net = VDN(C)
load_state_dict_cpu(net, checkpoint)

net.eval()

im_noisy = loadmat('/home/eunu/nas/DLA/test_data/DND/1.mat')['InoisySRGB']

H, W, _ = im_noisy.shape
if H % 2**dep_U != 0:
    H -= H % 2**dep_U
if W % 2**dep_U != 0:
    W -= W % 2**dep_U
im_noisy = im_noisy[:H, :W, ]

im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])



with torch.autograd.set_grad_enabled(False):
    tic = time.perf_counter()
    phi_Z = net(im_noisy, 'test')
    toc = time.perf_counter()
    err = phi_Z.cpu().numpy()

im_noisy = im_noisy.numpy()

print('Finish, time: {:.2f}'.format(toc-tic))
im_denoise = im_noisy - err[:, :C,]
im_denoise = np.transpose(im_denoise.squeeze(), (1,2,0))
im_denoise = img_as_ubyte(im_denoise.clip(0,1))
im_noisy = np.transpose(im_noisy.squeeze(), (1,2,0))
im_noisy = img_as_ubyte(im_noisy.clip(0,1))


plt.imsave('./results/real_test/im_noisy.jpg',im_noisy)
plt.imsave('./results/real_test/im_denoise.jpg',im_denoise)
