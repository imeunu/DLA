import numpy as np
import torch
import PIL.Image as pil_image 
from torch.utils.data import Dataset
import os 
import matplotlib.pyplot as plt 
import copy 

class Train(Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, data_type='float32', transform=None, sgm=25, ratio=0.9, size_data=(256, 256, 3), size_window=(5, 5)):
        self.data_dir = data_dir
        self.transform = transform
        self.data_type = data_type
        self.sgm = sgm

        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window

        lst_data = os.listdir(data_dir)

        # lst_input = [f for f in lst_data if f.startswith('input')]
        # lst_label = [f for f in lst_data if f.startswith('label')]
        #
        # lst_input.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        # lst_label.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        #
        # self.lst_input = lst_input
        # self.lst_label = lst_label

        lst_data.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
        # lst_data.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.lst_data = lst_data
        self.noise = self.sgm / 255.0 * np.random.randn(len(self.lst_data), self.size_data[0], self.size_data[1], self.size_data[2])

    def __getitem__(self, index):
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        # input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        #
        # if label.dtype == np.uint8:
        #     label = label / 255.0
        # if input.dtype == np.uint8:
        #     input = input / 255.0
        #
        # if label.ndim == 2:
        #     label = np.expand_dims(label, axis=2)
        # if input.ndim == 2:
        #     input = np.expand_dims(input, axis=2)
        #
        # if self.ny != label.shape[0]:
        #     label = label.transpose((1, 0, 2))
        # if self.ny != input.shape[0]:
        #     input = input.transpose((1, 0, 2))
        #
        # data = {'input': input, 'label': label}

        data = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))

        if data.dtype == np.uint8:
            data = data / 255.0

        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)

        if data.shape[0] > data.shape[1]:
            data = data.transpose((1, 0, 2))

        label = data + self.noise[index]
        input, mask = self.generate_mask(copy.deepcopy(label))

        data = {'label': label, 'input': input, 'mask': mask} # label : noised image, input : voided image, mask : mask 

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.lst_data)

    def generate_mask(self, input):

        ratio = self.ratio
        size_window = self.size_window
        size_data = self.size_data
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[2]):
            idy_msk = np.random.randint(0, size_data[0], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (idy_msk, idx_msk, ich)
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask