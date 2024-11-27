import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import numpy as np

def load_data_percentage(path_X, path_Y, path_Z, percentage=10):
    """
    Load a percentage of the data from the specified paths and normalize labels.
    
    :param path_X: str - Path to the X.npy file
    :param path_Y: str - Path to the Y.npy file
    :param path_Z: str - Path to the Z.npy file
    :param percentage: int - Percentage of data to load (0 < percentage <= 100)
    :return: tuple - (X, Y, Z) arrays with the specified percentage of data
    """
    # Load the full data
    X = np.load(path_X)
    Y = np.load(path_Y)
    Z = np.load(path_Z)
    
    # Calculate the number of samples to load
    num_samples = int(len(X) * (percentage / 100))
    
    # Slice the data to get the required percentage
    X = X[:num_samples]
    Y = Y[:num_samples]
    Z = Z[:num_samples]


    return X, Y, Z


class CustomDataset(Dataset):
    def __init__(self, X, Y, Z):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.Z = torch.tensor(Z, dtype=torch.float32).squeeze(-2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]

class SpecDataset(Dataset):
    def __init__(self, X, Y, Z, config, split_ratios=(0.7, 0.1, 0.2)):
        self.X, self.Y, self.Z = self.prepare_data(X, Y, Z, config)
        self.split_ratios = split_ratios
        self.split_data()

    def prepare_data(self, X, Y, Z, config):

        if config['model_type'] == 'resnet34':
            X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        elif config['model_type'] == 'resnet_attention':
            batch_size, length, dim = X.shape[0], X.shape[1], X.shape[2]
            X_tensor = X.reshape(batch_size, dim, length // 256, length // 128)
            X_tensor = torch.tensor(X_tensor, dtype=torch.float32)

        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        Z_tensor = torch.tensor(Z, dtype=torch.float32).squeeze(-2)

        return X_tensor, Y_tensor, Z_tensor

    def split_data(self):
        total_size = len(self.X)
        train_size = int(total_size * self.split_ratios[0])
        val_size = int(total_size * self.split_ratios[1])
        test_size = total_size - train_size - val_size

        self.train_indices, self.val_indices, self.test_indices = random_split(
            range(total_size), 
            [train_size, val_size, test_size]
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]

    def get_train_indices(self):
        return self.train_indices

    def get_val_indices(self):
        return self.val_indices

    def get_test_indices(self):
        return self.test_indices

def create_data_loaders(dataset, batch_size=32):
    train_indices = dataset.get_train_indices()
    val_indices = dataset.get_val_indices()
    test_indices = dataset.get_test_indices()

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader