import sys
import torch 
import torch.nn as nn 

from vanila.networks.VDN import VDN
import copy
from Unet.networks.UNet import UNet as pretrained_UNet

device = 'cuda:3'
checkpoint = torch.load('/home/eunu/nas/DLA/model_state_70.pth', map_location=device)

pre_model = nn.DataParallel(pretrained_UNet())
new_model = nn.DataParallel(VDN(3))

keys = [k[:7]+'DNet.'+k[7:] for k, v in pre_model.state_dict().items()]

new_model_dict = new_model.state_dict()
pre_trained_dict = {k: v for k, v in checkpoint.items() if k in keys[:-2]}
new_model_dict.update(pre_trained_dict)
new_model.load_state_dict(new_model_dict)

print(new_model)