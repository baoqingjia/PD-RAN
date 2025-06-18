import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import os

def load_spectra(filepath, shape):
    with open(filepath, 'r') as f:
        all_values = []
        for line in f:
            parts = line.strip().split()
            parts_converted = [p.replace('i', 'j') for p in parts]
            all_values.extend(parts_converted)

        complex_values = [complex(x) for x in all_values]

    return np.array(complex_values, dtype=np.complex64).reshape(shape)

def load_phase(filepath, shape):
    with open(filepath, 'r') as f:
        values = list(map(float, f.readline().split()))
    return np.array(values, dtype=np.float32).reshape(shape)

def split_dataset(dataset, train_ratio=0.9, val_ratio=0.1):
    total_size = len(dataset)
    print("total_size: ", total_size)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    print("train_size: ", train_size)
    print("val_size: ", val_size)
    return random_split(dataset, [train_size, val_size])

class TxtDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.path_spectra = os.path.join(root_dir, 'input_spectra')
        self.path_phase = os.path.join(root_dir, 'gt_phase')

        self.files = sorted(os.listdir(self.path_spectra))
        self.N = 1024 * 64
        self.iCount = len(self.files)

    def __len__(self):
        return self.iCount

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Load imag and imag part
        spectra = load_spectra(os.path.join(self.path_spectra, fname), (self.N,)) # 65536

        spectra_real = np.real(spectra)
        spectra_imag = np.imag(spectra)
        spectra = np.stack((spectra_real, spectra_imag), axis=0)
        spectra = spectra / np.abs(spectra).max()

        # Reshape to (2, 256, 256)
        spectra = spectra.reshape(2, 256, 256)
        spectra = torch.tensor(spectra, dtype=torch.float32) # (2, 256, 256)

        # Load gt_phase and convert to degrees
        phase = load_phase(os.path.join(self.path_phase, fname), (2,))
        phase = torch.tensor(phase, dtype=torch.float32) # (2,)

        return spectra, phase, fname


class TxtDataset_corrected(Dataset):
    def __init__(self, root_dir, transform=None):
        self.path_spectra = os.path.join(root_dir, 'input_spectra')
        self.path_phase = os.path.join(root_dir, 'gt_phase')

        self.files = sorted(os.listdir(self.path_spectra))
        self.N = 1024 * 64
        self.iCount = len(self.files)

    def __len__(self):
        return self.iCount

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Load imag and imag part
        spectra = load_spectra(os.path.join(self.path_spectra, fname), (self.N,)) # 65536

        spectra_real = np.real(spectra)
        spectra_imag = np.imag(spectra)
        spectra = np.stack((spectra_real, spectra_imag), axis=0)
        spectra = spectra / np.abs(spectra).max()

        # Reshape to (2, 256, 256)
        spectra = spectra.reshape(2, 256, 256)
        spectra = torch.tensor(spectra, dtype=torch.float32) # (2, 256, 256)

        return spectra, fname