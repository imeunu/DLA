import argparse
import os

import h5py
import torch

from networks.VDN import VDN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='/home/eunu/DLA/vanila/sigma_15/model_150.pth')
    parser.add_argument('--img_path', type=str, default='/home/junsung/DLA/test_data/CBSD68')
    parser.add_argument('--noise_type', type=str, default='awgn')
    parser.add_argument('--cuda', type=int, default=0)
    return parser.parse_args()

def load_model(args, device):
    model = VDN(3, )
    return

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)


if __name__ == '__main__':
    args = get_args()
    test(args)