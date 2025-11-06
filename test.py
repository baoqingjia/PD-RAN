"""Testing script for PD-RAN model

Evaluates trained PD-RAN model on test data and generates
comparison plots of phase-corrected spectra.
"""

import math
import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import PDRAN
from utils import TxtDataset
import torch.nn.functional as F, math

# Generate timestamp for logging
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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
    pred_vec = pred_vec / (pred_vec.norm(dim=1, keepdim=True) + 1e-8)
    gt_rad = gt_deg * math.pi / 180.0
    gt_vec = torch.stack([torch.cos(gt_rad), torch.sin(gt_rad)], dim=1)
    return F.mse_loss(pred_vec, gt_vec)

# Testing configuration (paths, device, batch size, epochs, etc.)
config = {
    'model_type': 'PDRAN',
    'current_time': current_time,
    'batch_size': 1,
    'cuda_device': torch.device("cuda:0"),
    'data_dir': 'data/samples/data_aug', # testing root
    'save_dir': 'checkpoint/vivo/best.pth',
    'results_dir': 'results/vivo/data_aug',
}

# Prepare logging directory/file
device = config['cuda_device']
log_dir = Path("log/test")
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f"{config['current_time']}_{config['model_type']}_test_log.txt"

# Dataset/loader
# TxtDataset returns: spectra tensor (2×256×256), phase tensor (2,), and filename.
test_dataset = TxtDataset(root_dir=config['data_dir'])
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

# Initialize model
model = PDRAN().to(device)

# Load trained model weights
if os.path.exists(config['save_dir']):
    model.load_state_dict(torch.load(config['save_dir'], map_location=device))
    print(f"Loaded model weights from {config['save_dir']}")
else:
    print("Warning: Model weights not found! Running with randomly initialized weights.")

# Initialization
criterion = nn.MSELoss()
model.eval()
test_loss = 0.0

sepc_list = []
phase_list = []
output_phase_list = []
fname_list = []

with open(log_file_path, "a") as log_file:
    with torch.no_grad():

        ph0_angle_mae_total = 0
        ph1_mae_total = 0
        total_samples = 0

        for batch_idx, (sepc, gt_phase, fnames) in enumerate(test_loader):
            sepc, gt_phase = sepc.to(device), gt_phase.to(device)

            # Forward pass → outputs: [B, 4] = [cosφ0, sinφ0, cosφ1, sinφ1]
            outputs = model(sepc)

            # Loss weights for the two heads (tunable hyperparameters)
            w1 = 10
            w2 = 10

            ph0_vec_pred = outputs[:, :2]
            ph1_vec_pred = outputs[:, 2:4]

            gt_ph0_deg = gt_phase[:, 0]
            gt_ph1_deg = gt_phase[:, 1]

            # Circular losses on (cos, sin) pairs for ph0 and ph1
            L_ph0 = ph0_vec_loss(ph0_vec_pred, gt_ph0_deg) * w1
            L_ph1 = ph0_vec_loss(ph1_vec_pred, gt_ph1_deg) * w2

            total_loss = L_ph0 + L_ph1

            test_loss += total_loss.item()

            # --- ph0 MAE with wrap-around: measure within [-180, 180)
            pred_ph0 = ph0_from_cos_sin(outputs[:, 0], outputs[:, 1])
            ph0_normalized = torch.remainder(pred_ph0 + 180, 360) - 180
            true_ph0 = gt_phase[:, 0]
            ph0_diff = torch.remainder(ph0_normalized - true_ph0 + 180, 360) - 180
            ph0_mae = ph0_diff.abs().sum()
            ph0_angle_mae_total += ph0_mae.item()

            # --- ph1 MAE (no wrap because typical range is narrower)
            pred_ph1 = ph0_from_cos_sin(outputs[:, 2], outputs[:, 3])
            true_ph1 = gt_phase[:, 1]
            ph1_mae = (pred_ph1 - true_ph1).abs().sum()
            ph1_mae_total += ph1_mae.item()

            total_samples += sepc.size(0)

            # Print and log every batch (set modulo to >1 to reduce verbosity)
            if (batch_idx + 1) % 1 == 0:
                # Convert tensors to NumPy
                pred_ph0_deg = ph0_normalized.detach().cpu().numpy()
                gt_ph0_deg = true_ph0.detach().cpu().numpy()

                pred_ph1_deg = pred_ph1.detach().cpu().numpy()
                gt_ph1_deg = true_ph1.detach().cpu().numpy()

                sample_display_count = min(2, pred_ph0_deg.shape[0])
                pred_ph0_display = ", ".join([f"{val:.2f}" for val in pred_ph0_deg[:sample_display_count]])
                gt_ph0_display = ", ".join([f"{val:.2f}" for val in gt_ph0_deg[:sample_display_count]])
                pred_ph1_display = ", ".join([f"{val:.2f}" for val in pred_ph1_deg[:sample_display_count]])
                gt_ph1_display = ", ".join([f"{val:.2f}" for val in gt_ph1_deg[:sample_display_count]])

                log_message = (
                    f"Step [{batch_idx + 1}/{len(test_loader)}], "
                    f"total_loss: {total_loss.item():.4f}, "
                    f"ph0_MAE(deg): {ph0_mae / sepc.size(0):.2f}, "
                    f"ph1_MAE(deg): {ph1_mae / sepc.size(0):.2f}, "
                    f"ph0_pred: [{pred_ph0_display}], "
                    f"ph0_true: [{gt_ph0_display}]"
                    f"ph1_pred: [{pred_ph1_display}], "
                    f"ph1_true: [{gt_ph1_display}]"
                    f"fnames: [{fnames}]"
                )

                print(log_message)
                log_file.write(log_message + "\n")
                log_file.flush()

            # Prepare arrays for plotting/saving
            pred_ph0_deg = ph0_normalized.detach().cpu().numpy()
            pred_ph1_deg = pred_ph1.detach().cpu().numpy()
            sepc = sepc.cpu().detach().numpy()
            gt_phase_all = gt_phase.cpu().detach().numpy()

            # Apply phase correction and generate comparison plots
            for i in range(sepc.shape[0]):
                pred_ph0 = pred_ph0_deg[i]
                pred_ph1 = pred_ph1_deg[i]

                gt_ph0, gt_ph1 = gt_phase_all[i, :2]

                N = 64 * 1024
                out_num = np.linspace(0, 1, N)
                pred_phase = (pred_ph0 + pred_ph1 * out_num) * np.pi / 180
                gt_phase = (gt_ph0 + gt_ph1 * out_num) * np.pi / 180

                # Extract real and imaginary parts
                sepc_real, sepc_imag = sepc[i, 0].flatten(), sepc[i, 1].flatten()

                # Apply phase corrections
                out_real = sepc_real * np.cos(-pred_phase) - sepc_imag * np.sin(-pred_phase)
                out_imag = sepc_real * np.sin(-pred_phase) + sepc_imag * np.cos(-pred_phase)
                gt_real = sepc_real * np.cos(-gt_phase) - sepc_imag * np.sin(-gt_phase)

                # Preserve filename
                fname = fnames[i]

                # Save real-part comparison figure: input / corrected / ground-truth
                os.makedirs(os.path.join(config['results_dir'], 'plot_spec'), exist_ok=True)
                img_filename = os.path.join(config['results_dir'], 'plot_spec', fname.split('.')[0] + '.jpg')
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].plot(sepc_real, color='red')
                axes[0].set_title("input_real")
                axes[1].plot(out_real, color='red')
                axes[1].set_title("corrected_real")
                axes[2].plot(gt_real, color='red')
                axes[2].set_title("gt_real")
                plt.tight_layout()

                plt.savefig(img_filename)
                plt.show()
                plt.close()

    # Final average test loss print and log
    avg_test_loss = test_loss / len(test_loader)
    avg_ph0_angle_mae = ph0_angle_mae_total / total_samples
    avg_ph1_mae = ph1_mae_total / total_samples

    final_log = (
        f"Average Test Loss: {avg_test_loss:.4f}\n"
        f"Average ph0_angle_mae: {avg_ph0_angle_mae:.4f}\n"
        f"Average ph1_mae: {avg_ph1_mae:.4f}"
    )
    
    print(final_log)