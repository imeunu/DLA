import sys
import torch 

from vanila.networks.UNet import UNet
import copy
from Unet.networks.UNet import UNet as pretrained_UNet

device = 'cuda:3'
checkpoint = torch.load('/home/eunu/nas/DLA/model_state_70.pth', map_location=device)

pre_model = pretrained_UNet()
new_model = UNet()
#----------for data parallel error---------# 
from collections import OrderedDict

new_state_dict =OrderedDict()
for k, v in checkpoint.items():
    name = k[7:]
    new_state_dict[name] = v
#-----------------------------------------#

keys = [k for k, v in new_state_dict.items()]

new_model_dict = new_model.state_dict()
pre_trained_dict = {k: v for k, v in new_state_dict.items() if k in keys[:-2]}
new_model_dict.update(pre_trained_dict)
new_model.load_state_dict(new_model_dict)


print(new_model)
