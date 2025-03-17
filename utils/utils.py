import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import h5py
import numpy as np


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    total_size = len(dataset)
    print("total_size: ", total_size)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    print("test_size: ", test_size)
    return random_split(dataset, [train_size, val_size, test_size])


class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform  

        with h5py.File(self.file_path, "r") as f:
            self.data_len = f["data_rm"].shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as f:
            data_rm = f["data_rm"][idx]
            gt_real = f["gt_real"][idx]
            gt_phase = f["gt_phase"][idx]

        spec, channels = data_rm.shape[0], data_rm.shape[1]
        data_rm = data_rm.transpose(1, 0)
        data_rm = data_rm.reshape(channels, 256, 256)

        data_rm = torch.tensor(data_rm, dtype=torch.float32)
        gt_real = torch.tensor(gt_real, dtype=torch.float32)
        gt_phase = torch.tensor(gt_phase, dtype=torch.float32).squeeze(0)
        
        if self.transform:
            data_rm = self.transform(data_rm)

        return data_rm, gt_real, gt_phase