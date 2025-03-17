from model.model import PDRAN
import torch
from utils.utils import HDF5Dataset, split_dataset
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn 
import torch
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Subset

now = datetime.datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

config = {
    'model_type': 'PDRAN',
    'current_time': current_time,
    'epoch': 400,
    'batch_size': 16,
    'cuda_device': torch.device("cuda:0"),
    'data_dir': 'data/simu/train_100.h5',
    'save_dir': 'checkpoint/simu/'
}

os.makedirs('log/train/', exist_ok=True)
os.makedirs(config['save_dir'], exist_ok=True)

dataset = HDF5Dataset(config['data_dir'])

train_data, val_data, test_data = split_dataset(dataset)

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=8)
val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=8)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers=8)

device = config['cuda_device']
num_epochs = config['epoch']

model = PDRAN().to(device)

best_val_loss = float('inf')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

criterion = nn.MSELoss()

patience = 5
early_stop_counter = 0

log_file_path = f"log/train/{config['current_time']}_{config['model_type']}_train_log.txt"

with open(log_file_path, "a") as log_file:
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0
        for batch_idx, (x, _, z) in enumerate(train_loader):
            x, z = x.to(device), z.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, z)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _, z in val_loader:
                x, z = x.to(device), z.to(device)
                outputs = model(x)
                loss = criterion(outputs, z)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        log_message = (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, '
                       f'Val Loss: {avg_val_loss:.4f}')
        print(log_message)
        log_file.write(log_message + "\n")
        log_file.flush()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_name = f"{config['save_dir']}/{config['current_time']}_{config['model_type']}_best.pth"
            torch.save(model.state_dict(), best_model_name)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            log_file.write(f"Saved best model with validation loss: {best_val_loss:.4f}\n")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter}/{patience} epochs.")
            log_file.write(f"No improvement in validation loss for {early_stop_counter}/{patience} epochs.\n")

        if early_stop_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            log_file.write("Early stopping triggered. Training stopped.\n")
            break

print("Training finished!")