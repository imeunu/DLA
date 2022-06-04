import h5py
import numpy as np
import torch

from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, args):
        self.args = args
        with h5py.File(f'{args.h5_path}', 'r') as f:
            self.images = np.array(f['train'])

    def __getitem__(self, idx):
        clean = torch.Tensor(self.images[idx])
        if self.args.strategy == 'std':
            noisy = torch.Tensor(self.add_noise(clean))
            return (noisy, clean)

        elif self.args.strategy == 'n2n':
            noisy = torch.Tensor(self.add_noise(clean))
            target = torch.Tensor(self.add_noise(clean))
            return (noisy, target)

        elif self.args.strategy == 'n2v':
            return ()

    def __len__(self):
        return len(self.images)
    
    def add_noise(self, img, type='Gaussian'):
        if type == 'Gaussian':
            img_size = np.shape(img)
            noise = np.random.normal(0, self.args.sigma, img_size)
            noisy = img + noise
        return np.clip(noisy, 0, 255)

# np.random.seed
'''
noisy = clean + noise
return (noisy, clean)
'''