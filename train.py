"""Training script for PD-RAN model

Trains the PD-RAN model for NMR phase correction with early stopping.
"""

from model import PDRAN
from utils import split_dataset, TxtDataset
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch
import datetime
import os

# Generate timestamp for logging and model saving
now = datetime.datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

# Training configuration
config = {
    'model_type': 'PDRAN',
    'current_time': current_time,
    'epoch': 400,
    'batch_size': 4,
    'cuda_device': torch.device("cuda:0"),
    'data_dir': 'data/simu/train', # 'vivo' or 'simu'
    'save_dir': 'checkpoint/simu/' # 'vivo' or 'simu'
}

# Create directories for logs and checkpoints
os.makedirs('log/train/', exist_ok=True)
os.makedirs(config['save_dir'], exist_ok=True)

# Load and split dataset
dataset = TxtDataset(root_dir=config['data_dir'])
train_data, val_data = split_dataset(dataset)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=8)
val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=8)

# Initialize training components
device = config['cuda_device']
num_epochs = config['epoch']
model = PDRAN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# Early stopping parameters
best_val_loss = float('inf')
patience = 30
early_stop_counter = 0

log_file_path = f"log/train/{config['current_time']}_{config['model_type']}_train_log.txt"

with open(log_file_path, "a") as log_file:
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (sepc, phase, _) in enumerate(train_loader):
            sepc, phase = sepc.to(device), phase.to(device)
            optimizer.zero_grad()
            outputs = model(sepc)
            loss = criterion(outputs, phase)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sepc, phase, _ in val_loader:
                sepc, phase = sepc.to(device), phase.to(device)
                outputs = model(sepc)
                loss = criterion(outputs, phase)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Log training progress
        log_message = (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, '
                       f'Val Loss: {avg_val_loss:.4f}')
        print(log_message)
        log_file.write(log_message + "\n")
        log_file.flush()

        # Save best model and early stopping logic
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