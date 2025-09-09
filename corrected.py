"""
PD-RAN Phase Correction Inference Script

Applies trained PD-RAN model to correct phase errors in NMR spectra.
Outputs corrected real-part spectra and predicted phase parameters.
"""

import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model import PDRAN
from utils import TxtDataset_corrected

# Generate timestamp for logging
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Configuration parameters
config = {
    'model_type': 'PDRAN',
    'current_time': current_time,
    'batch_size': 4,
    'cuda_device': torch.device("cuda:0"),
    'data_dir': 'data/simu/test/',  # Input test data directory, vivo: data/vivo/test/data_aug/ or data/vivo/test/data_ori/, simu: data/simu/test/
    'save_dir': 'checkpoint/simu/best.pth',  # Trained model weights, 'vivo' or 'simu'
    'results_dir': 'results/simu/', # Output results directory, vivo: results/vivo/data_aug/ or results/vivo/data_ori/, simu: results/simu/
}

# Setup device and logging
device = config['cuda_device']
log_dir = Path("log/test")
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f"{config['current_time']}_{config['model_type']}_test_log.txt"
last_dir = os.path.basename(os.path.normpath(config['results_dir']))

# Load test dataset
test_dataset = TxtDataset_corrected(root_dir=config['data_dir'])
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                        shuffle=False, num_workers=8, 
                        pin_memory=torch.cuda.is_available())

# Initialize and load trained model
model = PDRAN().to(device)

if os.path.exists(config['save_dir']):
    model.load_state_dict(torch.load(config['save_dir'], map_location=device))
    print(f"Loaded model weights from {config['save_dir']}")
else:
    print("Warning: Model weights not found! Running with randomly initialized weights.")

criterion = nn.MSELoss()

# Set model to evaluation mode
model.eval()
test_loss = 0.0

# Initialize storage lists
sepc_list = []
phase_list = []
output_phase_list = []
fname_list = []

# Process test data and apply phase correction
with open(log_file_path, "a") as log_file:
    with torch.no_grad():
        for batch_idx, (sepc, fnames) in enumerate(test_loader):
            sepc = sepc.to(device)

            # Predict phase parameters
            output_phase = model(sepc)

            # Convert to NumPy for processing
            output_phase = output_phase.cpu().detach().numpy()
            sepc = sepc.cpu().detach().numpy()

            # Apply phase correction to each spectrum
            for i in range(sepc.shape[0]):
                # Extract predicted phase parameters
                out_ph0, out_ph1 = output_phase[i, :2]

                # Generate phase correction array
                N = 64 * 1024
                out_num = np.linspace(0, 1, N)
                out_phase = (out_ph0 + out_ph1 * out_num) * np.pi / 180

                # Apply phase correction to complex spectrum
                sepc_real, sepc_imag = sepc[i, 0].flatten(), sepc[i, 1].flatten()

                if last_dir == "vivo":
                    out_real = sepc_real * np.cos(-out_phase) - sepc_imag * np.sin(-out_phase)
                    out_imag = sepc_real * np.sin(-out_phase) + sepc_imag * np.cos(-out_phase)
                elif last_dir == "simu":
                    out_real = sepc_real * np.cos(out_phase) - sepc_imag * np.sin(out_phase)
                    out_imag = sepc_real * np.sin(out_phase) + sepc_imag * np.cos(out_phase)

                # Save corrected spectrum and phase parameters
                fname = fnames[i]

                # Save corrected real-part spectrum
                real_spectra_dir = os.path.join(config['results_dir'], 'corrected_real_part')
                os.makedirs(real_spectra_dir, exist_ok=True)
                real_txt_path = os.path.join(real_spectra_dir, fname)
                np.savetxt(real_txt_path, out_real.reshape(-1, 1), fmt='%.6f')

                # Save predicted phase parameters
                os.makedirs(os.path.join(config['results_dir'], 'predicted_phase'), exist_ok=True)
                phase_path = os.path.join(config['results_dir'], 'predicted_phase', fname)
                np.savetxt(phase_path, np.column_stack((out_ph0, out_ph1)), delimiter=' ', fmt='%.6f')