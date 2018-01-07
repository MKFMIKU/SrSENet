import torch.utils.data as data
import torch
import numpy as np
import h5py


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("input")
        self.label_x2 = hf.get("hr_x2")
        self.label_x4 = hf.get("hr_x4")
        self.label_x8 = hf.get("hr_x8")
        self.bicubic_x2 = hf.get("bicubic_x2")
        self.bicubic_x4 = hf.get("bicubic_x4")
        self.bicubic_x8 = hf.get("bicubic_x8")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index, :, :, :]).float(), \
               torch.from_numpy(self.label_x2[index, :, :, :]).float(), \
               torch.from_numpy(self.label_x4[index, :, :, :]).float(), \
               torch.from_numpy(self.label_x8[index, :, :, :]).float(), \
               torch.from_numpy(self.bicubic_x2[index, :, :, :]).float(), \
               torch.from_numpy(self.bicubic_x4[index, :, :, :]).float(), \
               torch.from_numpy(self.bicubic_x8[index, :, :, :]).float()

    def __len__(self):
        return self.data.shape[0]
