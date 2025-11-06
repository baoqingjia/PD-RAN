"""Utility functions and dataset classes for PD-RAN

Provides data loading, preprocessing, and dataset splitting functionality
for NMR spectroscopy data.
"""

import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import os

# Load a complex spectrum from a text file. The file is expected to contain values
# where imaginary unit may be written as 'i'; this is replaced with 'j' to satisfy
# Python's complex literal. The returned array is truncated to 65536 samples (64k).
def load_spectra(filepath, shape):
    """Load complex NMR spectra from text file"""
    with open(filepath, 'r') as f:
        all_values = []
        for line in f:
            parts = line.strip().split()
            parts_converted = [p.replace('i', 'j') for p in parts]
            all_values.extend(parts_converted)

        complex_values = [complex(x) for x in all_values]

    # Optionally check size mismatch here; kept commented to avoid noisy logs.
    # if np.array(complex_values).size != 65536:
    #   print(filepath, np.array(complex_values).size)

    return np.array(complex_values, dtype=np.complex64)[:65536]

# Load phase parameters (two floats per file): [ph0_deg, ph1_deg]
# `shape` is ignored except for reshaping to (2,) downstream.
def load_phase(filepath, shape):
    """Load phase parameters from text file"""
    with open(filepath, 'r') as f:
        values = list(map(float, f.readline().split()))
    return np.array(values, dtype=np.float32).reshape(shape)

# Deterministic train/val split with seed control. Ensures both sets are non-empty
# Returns two Subset objects.
# The random split uses a fixed Generator with the given seed for reproducibility.
def split_dataset(dataset, train_ratio: float = 0.9, val_ratio: float = 0.1, seed: int = 3407):
    """Split dataset into training and validation sets"""
    n = len(dataset)
    print("total_size:", n)

    if n == 0:
        raise ValueError("Empty dataset.")

    # If ratios don't sum to 1, normalize to avoid rounding inconsistencies
    s = train_ratio + val_ratio
    if s <= 0:
        raise ValueError("train_ratio + val_ratio must be > 0.")
    train_ratio = train_ratio / s
    
    n_train = int(n * train_ratio)
    n_val = n - n_train # complement to ensure n_train + n_val == n

    # Small-sample protection: keep both splits non-empty when possible
    if n >= 2:
        if n_train == 0:
            n_train, n_val = 1, n - 1
        elif n_val == 0:
            n_val, n_train = 1, n - 1

    print("train_size:", n_train)
    print("val_size:", n_val)

    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=g)

class TxtDataset(Dataset):
    """Dataset class for loading NMR spectra and phase parameters"""
    def __init__(self, root_dir, transform=None):
        # Expects structure:
        # root_dir/
        #   input_spectra/ (text complex spectra)
        #   gt_phase/ (text phase: two floats per file)
        self.path_spectra = os.path.join(root_dir, 'input_spectra')
        self.path_phase = os.path.join(root_dir, 'gt_phase')

        self.files = sorted(os.listdir(self.path_spectra))
        self.N = 1024 * 64 # number of points per spectrum (64k)
        self.iCount = len(self.files)

    def __len__(self):
        return self.iCount

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Load complex spectrum and split into real/imag channels
        spectra = load_spectra(os.path.join(self.path_spectra, fname), (self.N,)) # 65536

        spectra_real = np.real(spectra)
        spectra_imag = np.imag(spectra)
        spectra = np.stack((spectra_real, spectra_imag), axis=0)
        spectra = spectra / np.abs(spectra).max() # per-sample normalization to [-1,1]

        # Reshape 1D 64k into 2D 256×256 for CNN input
        spectra = spectra.reshape(2, 256, 256)
        spectra = torch.tensor(spectra, dtype=torch.float32) # (2, 256, 256)

        # Load ground-truth phase [ph0_deg, ph1_deg]
        phase = load_phase(os.path.join(self.path_phase, fname), (2,))
        phase = torch.tensor(phase, dtype=torch.float32) # (2,)

        return spectra, phase, fname


class TxtDataset_corrected(Dataset):
    """Dataset class for loading spectra without phase parameters (for inference)"""
    def __init__(self, root_dir, transform=None):
        # Like TxtDataset, but returns spectra and filename only (no GT phase).
        self.path_spectra = os.path.join(root_dir, 'input_spectra')

        self.files = sorted(os.listdir(self.path_spectra))
        self.N = 1024 * 64
        self.iCount = len(self.files)

    def __len__(self):
        return self.iCount

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Load and normalize complex spectrum as (2, 256, 256)
        spectra = load_spectra(os.path.join(self.path_spectra, fname), (self.N,)) # 65536

        spectra_real = np.real(spectra)
        spectra_imag = np.imag(spectra)
        spectra = np.stack((spectra_real, spectra_imag), axis=0)
        spectra = spectra / np.abs(spectra).max()

        # Reshape to (2, 256, 256)
        spectra = spectra.reshape(2, 256, 256)
        spectra = torch.tensor(spectra, dtype=torch.float32) # (2, 256, 256)

        return spectra, fname