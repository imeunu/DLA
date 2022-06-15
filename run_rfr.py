import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from VDN.model import VDN


def load_model(args, device):
    model = nn.DataParallel(VDN(3, wf=64, dep_U=4))
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    return model

def run_rfr(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rfr_ckpt = f'/home/eunu/DLA/RFR/'
    os.makedirs(rfr_ckpt, exist_ok=True)
    model = load_model(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    torch.save(optimizer.state_dict(), f'{args.ckpt_path}/opt.pth')

    for epoch in range(args.iters):


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--iters', type=int, default=40)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    run_rfr(args)

'''
augmentation
M = 10, 20, 40
'''