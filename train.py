"""Training script for PD-RAN model

Trains the PD-RAN model for NMR phase correction with early stopping.
"""

import math
from model import PDRAN
from utils import split_dataset, TxtDataset
from torch.utils.data import DataLoader
import torch
import datetime
import os
import torch.nn.functional as F, math

# Generate timestamp for logging and model saving
now = datetime.datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

# (cos, sin) → angle in degrees within [-180, 180)
def ph0_from_cos_sin(cos_phi0, sin_phi0, normalize=True):
    if normalize:
        v = torch.stack([cos_phi0, sin_phi0], dim=-1)
        v = F.normalize(v, p=2, dim=-1)
        cos_phi0, sin_phi0 = v[...,0], v[...,1]

    phi_rad = torch.atan2(sin_phi0, cos_phi0)

    phi_deg = torch.rad2deg(phi_rad)
    phi_deg = torch.remainder(phi_deg + 180.0, 360.0) - 180.0

    return phi_deg

# Unit-circle MSE between predicted (cos, sin) and GT vector from degrees
# Robust to wrap-around of angles; encourages predictions on the circle.
def ph0_vec_loss(pred_vec, gt_deg):
    # pred_vec: [B,2] -> normalize to unit circle
    pred_vec = pred_vec / (pred_vec.norm(dim=1, keepdim=True) + 1e-8)
    gt_rad = gt_deg * math.pi / 180.0
    gt_vec = torch.stack([torch.cos(gt_rad), torch.sin(gt_rad)], dim=1)
    return F.mse_loss(pred_vec, gt_vec)

# Training configuration (paths, device, batch size, epochs, etc.)
config = {
    'model_type': 'PDRAN',
    'current_time': current_time,
    'epoch': 400,
    'batch_size': 16,
    'cuda_device': torch.device("cuda:0"),
    'data_dir': 'data/vivo/train', # training root
    'save_dir': 'checkpoint/vivo/' # output folder for checkpoints'
}

# Create directories for logs and checkpoints
os.makedirs('log/train/', exist_ok=True)
os.makedirs(config['save_dir'], exist_ok=True)

# Build dataset and deterministic train/val split
# TxtDataset expects each case to have a complex spectrum (as text) and GT phase.
dataset = TxtDataset(root_dir=config['data_dir'])
train_data, val_data = split_dataset(dataset)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=8)
val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=8)

# Initialize training components
device = config['cuda_device']
num_epochs = config['epoch']
model = PDRAN().to(device)
best_val_loss = float('inf')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Early stopping if val loss does not improve for `patience` epochs
patience = 30
early_stop_counter = 0

log_file_path = f"log/train/{config['current_time']}_{config['model_type']}_train_log.txt"

with open(log_file_path, "a") as log_file:
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        ph0_angle_mae_total = 0
        ph1_mae_total = 0

        for batch_idx, (sepc, gt_phase, _) in enumerate(train_loader):
            # sepc: (B,2,256,256), gt_phase: (B,2) with [ph0_deg, ph1_deg]
            sepc, gt_phase = sepc.to(device), gt_phase.to(device)
            optimizer.zero_grad()

            outputs = model(sepc) # outputs: (B,4) = [cosφ0, sinφ0, cosφ1, sinφ1]

            # Loss weights for the two heads (tunable hyperparameters)
            w1 = 10
            w2 = 10

            ph0_vec_pred = outputs[:, :2]
            ph1_vec_pred = outputs[:, 2:4]

            gt_ph0_deg = gt_phase[:, 0]
            gt_ph1_deg = gt_phase[:, 1]

            # Weighted circular losses for two heads
            L_ph0 = ph0_vec_loss(ph0_vec_pred, gt_ph0_deg) * w1 # ph0 loss
            L_ph1 = ph0_vec_loss(ph1_vec_pred, gt_ph1_deg) * w2 # ph1 loss

            total_loss = L_ph0 + L_ph1

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

            # --- ph0 MAE with wrap-around to avoid 360° boundary artifacts
            pred_ph0 = ph0_from_cos_sin(outputs[:, 0], outputs[:, 1])
            ph0_normalized = torch.remainder(pred_ph0 + 180, 360) - 180
            true_ph0 = gt_phase[:, 0]
            ph0_diff = torch.remainder(ph0_normalized - true_ph0 + 180, 360) - 180
            ph0_mae = ph0_diff.abs().sum()
            ph0_angle_mae_total += ph0_mae.item()

            # --- ph1 MAE (commonly narrow range; wrap not applied here)
            pred_ph1 = ph0_from_cos_sin(outputs[:, 2], outputs[:, 3])
            ph1_mae = (pred_ph1 - gt_phase[:, 1]).abs().sum()
            ph1_mae_total += ph1_mae.item()

            # Periodic progress printing
            if (batch_idx + 1) % 10 == 0:
                pred_ph0_deg = ph0_normalized.detach().cpu().numpy()
                gt_ph0_deg = true_ph0.detach().cpu().numpy()

                sample_display_count = min(2, pred_ph0_deg.shape[0])
                pred_ph0_display = ", ".join([f"{val:.2f}" for val in pred_ph0_deg[:sample_display_count]])
                gt_ph0_display = ", ".join([f"{val:.2f}" for val in gt_ph0_deg[:sample_display_count]])

                batch_size = sepc.size(0)

                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Step [{batch_idx + 1}/{len(train_loader)}], '
                    f'Loss: {total_loss.item() :.4f}, '
                    f'L1_ph0: {L_ph0.item() :.4f}, '
                    f'L2_ph1: {L_ph1.item() :.4f}, '
                    f'ph0_MAE(deg): {ph0_mae.item() / batch_size:.2f}, '
                    f'ph1_MAE(deg): {ph1_mae.item() / batch_size:.2f}, '
                    f'ph0_pred: [{pred_ph0_display}], '
                    f'ph0_true: [{gt_ph0_display}]')


        avg_train_loss = train_loss / len(train_loader)

        total_train_samples = len(train_loader.dataset)
        avg_train_ph0_mae = ph0_angle_mae_total / total_train_samples
        avg_train_ph1_mae = ph1_mae_total / total_train_samples

        # Validation phase
        model.eval()
        val_loss = 0
        val_ph0_mae_total = 0
        val_ph1_mae_total = 0
        val_L1loss_ph0_total = 0
        val_L2loss_ph1_total = 0

        with torch.no_grad():
            for sepc, gt_phase, _ in val_loader:
                sepc, gt_phase = sepc.to(device), gt_phase.to(device)
                outputs = model(sepc)

                ph0_vec_pred = outputs[:, :2]
                ph1_vec_pred = outputs[:, 2:4]

                gt_ph0_deg = gt_phase[:, 0]
                gt_ph1_deg = gt_phase[:, 1]

                # Use same weights as training
                L_ph0 = ph0_vec_loss(ph0_vec_pred, gt_ph0_deg) * w1
                L_ph1 = ph0_vec_loss(ph1_vec_pred, gt_ph1_deg) * w2

                total_loss = L_ph0 + L_ph1

                val_L1loss_ph0_total += L_ph0.item()
                val_L2loss_ph1_total += L_ph1.item()
                val_loss += total_loss.item()

                # ph0 MAE with wrapping
                pred_ph0 = ph0_from_cos_sin(outputs[:, 0], outputs[:, 1])
                ph0_normalized = torch.remainder(pred_ph0 + 180, 360) - 180
                true_ph0 = gt_phase[:, 0]
                ph0_diff = torch.remainder(ph0_normalized - true_ph0 + 180, 360) - 180
                val_ph0_mae_total += ph0_diff.abs().sum().item()

                # ph1 MAE without wrap
                pred_ph1 = ph0_from_cos_sin(outputs[:, 2], outputs[:, 3])
                val_ph1_mae_total += (pred_ph1 - gt_phase[:, 1]).abs().sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_L1loss_ph0 = val_L1loss_ph0_total / len(val_loader)
        avg_L2loss_ph1 = val_L2loss_ph1_total / len(val_loader)

        total_val_samples = len(val_loader.dataset)
        avg_val_ph0_mae = val_ph0_mae_total / total_val_samples
        avg_val_ph1_mae = val_ph1_mae_total / total_val_samples

        log_message = (f'Epoch [{epoch+1}/{num_epochs}] | '
                       f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | '
                       f'avg_L1loss_ph0: {avg_L1loss_ph0:.4f}, '
                       f'avg_L2loss_ph1: {avg_L2loss_ph1:.4f}, '
                       f'Train ph0_MAE: {avg_train_ph0_mae:.2f} | Val ph0_MAE: {avg_val_ph0_mae:.2f} | '
                       f'Train ph1_MAE: {avg_train_ph1_mae:.2f} | Val ph1_MAE: {avg_val_ph1_mae:.2f}')
        
        print(log_message)
        log_file.write(log_message + "\n")
        log_file.flush()

        # Save best model by val loss
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