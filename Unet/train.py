from glob import glob
import warnings
import time
import random
import numpy as np
import shutil
import torchvision.utils as vutils
from utils import batch_PSNR, batch_SSIM
from torch.utils.tensorboard import SummaryWriter
from math import ceil
from networks.VDN import UNet, weight_init_kaiming
from dataset import SimulateH5, SimulateTest
import torch.utils.data as uData
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
from pathlib import Path
from options import set_opts

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

args = set_opts()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if isinstance(args.gpu_id, int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))

_C = 3
_lr_min = 1e-6
_modes = ['train', 'test_cbsd681', 'test_cbsd682', 'test_cbsd683']

def train_model(net, datasets, optimizer, lr_scheduler, criterion, sigma):
    os.makedirs(f'{args.model_dir}/sigma_{sigma}', exist_ok=True)
    clip_grad_D = args.clip_grad_D
    clip_grad_S = args.clip_grad_S
    batch_size = {'train': args.batch_size, 'test_cbsd681': 1, 'test_cbsd682': 1, 'test_cbsd683': 1}

    train_loader = uData.DataLoader(datasets['train'], batch_size=batch_size['train'], shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test1_loader = uData.DataLoader(datasets['test_cbsd681'], batch_size=batch_size['test_cbsd681'], shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test2_loader = uData.DataLoader(datasets['test_cbsd682'], batch_size=batch_size['test_cbsd682'], shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test3_loader = uData.DataLoader(datasets['test_cbsd683'], batch_size=batch_size['test_cbsd683'], shuffle=False, num_workers=args.num_workers, pin_memory=True)

    data_loader = {'train' : train_loader, 'test_cbsd681' : test1_loader, 'test_cbsd682' : test2_loader, 'test_cbsd683' : test3_loader}

    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in datasets.keys()}
    writer = SummaryWriter(f'{args.log_dir}/sigma_{sigma}')
    if args.resume:
        step = args.step
        step_img = args.step_img
    else:
        step = 0
        step_img = {x: 0 for x in _modes}
    param_D = [x for name, x in net.named_parameters() if 'dnet' in name.lower()]
    param_S = [x for name, x in net.named_parameters() if 'snet' in name.lower()]
    for epoch in range(args.epoch_start, args.epochs):
        loss_per_epoch = {x: 0 for x in ['Loss', 'lh', 'KLG', 'KLIG']}
        mse_per_epoch = {x: 0 for x in _modes}
        grad_norm_D = grad_norm_S = 0
        tic = time.time()
        # train stage
        net.train()
        lr = optimizer.param_groups[0]['lr']
        #if lr < _lr_min:
        #    sys.exit('Reach the minimal learning rate')
        phase = 'train'
        for ii, data in enumerate(data_loader[phase]):
            im_noisy, im_gt = [x.cuda() for x in data]
            optimizer.zero_grad()
            phi_Z = net(im_noisy)
            loss = criterion(im_gt, phi_Z)
            loss.backward()
            optimizer.step()
            if not ii % 40:
                print(f'Epoch: {epoch}, iteration: {ii}, loss: {loss}')
        lr_scheduler.step()
        # save model
        if (epoch+1) % args.save_model_freq == 0 or epoch+1 == args.epochs:
            model_prefix = 'model_'
            save_path_model = os.path.join(f'{args.model_dir}/sigma_{sigma}', model_prefix+str(epoch+1)+'.pth')
            torch.save({
                'epoch': epoch+1,
                'step': step+1,
                'step_img': {x: step_img[x] for x in _modes},
                'grad_norm_D': clip_grad_D,
                'grad_norm_S': clip_grad_S,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, save_path_model)
            model_state_prefix = 'model_state_'
            save_path_model_state = os.path.join(f'{args.model_dir}/sigma_{sigma}', model_state_prefix+str(epoch+1)+'.pth')
            torch.save(net.state_dict(), save_path_model_state)

        # writer.add_scalars('MSE_epoch', mse_per_epoch, epoch)
        # writer.add_scalars('PSNR_epoch_test', psnr_per_epoch, epoch)
        # writer.add_scalars('SSIM_epoch_test', ssim_per_epoch, epoch)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')

def main(sigma):
    # build the model
    net = UNet(_C, slope=args.slope, wf=args.wf, depth=args.depth)
    # move the model to GPU
    net = nn.DataParallel(net).cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # args.milestones = [20, 70, 150, 300, 500]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma = args.gamma)

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> Loading checkpoint {:s}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.epoch_start = checkpoint['epoch']
            args.step = checkpoint['step']
            args.step_img = checkpoint['step_img']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            net.load_state_dict(checkpoint['model_state_dict'])
            args.clip_grad_D = checkpoint['grad_norm_D']
            args.clip_grad_S = checkpoint['grad_norm_S']
            print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args.resume, checkpoint['epoch']))
        else:
            sys.exit('Please provide corrected model path!')
    else:
        net = weight_init_kaiming(net)
        args.epoch_start = 0
        if os.path.isdir(f'{args.log_dir}/sigma_{sigma}'):
            shutil.rmtree(f'{args.log_dir}/sigma_{sigma}')
        os.makedirs(f'{args.log_dir}/sigma_{sigma}')
        if os.path.isdir(f'{args.log_dir}/sigma_{sigma}'):
            shutil.rmtree(f'{args.log_dir}/sigma_{sigma}')
        os.makedirs(f'{args.log_dir}/sigma_{sigma}')

    # print the arg pamameters
    for arg in vars(args):
        print('{:<15s}: {:s}'.format(arg,  str(getattr(args, arg))))

    # making traing data
    simulate_dir = Path(args.simulate_dir)
    train_im_list = list(simulate_dir.glob('*.jpg')) + list(simulate_dir.glob('*.png')) + \
                                                                    list(simulate_dir.glob('*.bmp'))
    train_im_list = sorted([str(x) for x in train_im_list])
    # making tesing data
    test_case1_h5 = Path('/home/junsung/DLA/test_data').joinpath('noise_niid', 'CBSD68_niid_case1.hdf5')
    test_case2_h5 = Path('/home/junsung/DLA/test_data').joinpath('noise_niid', 'CBSD68_niid_case2.hdf5')
    test_case3_h5 = Path('/home/junsung/DLA/test_data').joinpath('noise_niid', 'CBSD68_niid_case3.hdf5')
    test_im_list = (Path('/home/junsung/DLA/test_data') / 'CBSD68').glob('*.png')
    test_im_list = sorted([str(x) for x in test_im_list])
    datasets = {'train': SimulateH5(h5_path = args.simulateh5_dir, 
                                          pch_size = args.patch_size, radius=args.radius, sigma=sigma),
                         'test_cbsd681':SimulateTest(test_im_list, test_case1_h5),
                        'test_cbsd682': SimulateTest(test_im_list, test_case2_h5),
                        'test_cbsd683': SimulateTest(test_im_list, test_case3_h5)}
    # train model
    print('\nBegin training with GPU: ' + str(args.gpu_id))
    train_model(net, datasets, optimizer, scheduler, nn.MSELoss(), sigma)

if __name__ == '__main__':
    for sigma in [25]:
        main(sigma)