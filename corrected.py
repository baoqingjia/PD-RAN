"""
PD-RAN Phase Correction Inference Script

Applies trained PD-RAN model to correct phase errors in NMR spectra.
Outputs corrected real-part spectra and predicted phase parameters.
"""

import os
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model import PDRAN
from utils import TxtDataset_corrected
import torch, torch.nn.functional as F

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

# Correcting configuration (paths, device, batch size, epochs, etc.)
config = {
    'model_type': 'PDRAN',
    'current_time': current_time,
    'batch_size': 1,
    'cuda_device': torch.device("cuda:0"),
    'data_dir': 'data/samples/data_aug',
    'save_dir': 'checkpoint/vivo/best.pth',
    'results_dir': 'results/vivo/data_aug',
}

device = config['cuda_device']

# Load test dataset
test_dataset = TxtDataset_corrected(root_dir=config['data_dir'])
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

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
with torch.no_grad():
    for batch_idx, (sepc, fnames) in enumerate(test_loader):
        sepc = sepc.to(device)

        # Predict phase parameters
        outputs = model(sepc)

        # Convert to NumPy for processing
        sepc = sepc.cpu().detach().numpy()

        # Process each sample in batch
        for i in range(sepc.shape[0]):

            pred_ph0 = ph0_from_cos_sin(outputs[:, 0], outputs[:, 1])
            pred_ph1 = ph0_from_cos_sin(outputs[:, 2], outputs[:, 3])

            pred_ph0 = pred_ph0.cpu().detach().numpy()
            pred_ph1 = pred_ph1.cpu().detach().numpy()

            N = 64 * 1024
            out_num = np.linspace(0, 1, N) # 
            pred_phase = (pred_ph0 + pred_ph1 * out_num) * np.pi / 180 

            sepc_real, sepc_imag = sepc[i, 0].flatten(), sepc[i, 1].flatten()

            out_real = sepc_real * np.cos(-pred_phase) - sepc_imag * np.sin(-pred_phase)
            out_imag = sepc_real * np.sin(-pred_phase) + sepc_imag * np.cos(-pred_phase)

            # Get the corresponding filename
            fname = fnames[i]

            # Create directory for real part input_spectra if not exists
            real_spectra_dir = os.path.join(config['results_dir'], 'corrected_real_part')
            os.makedirs(real_spectra_dir, exist_ok=True)

            # Save corrected real-part spectrum
            real_txt_path = os.path.join(real_spectra_dir, fname)
            np.savetxt(real_txt_path, out_real.reshape(-1, 1), fmt='%.6f')

            # Save predicted phase parameters
            os.makedirs(os.path.join(config['results_dir'], 'predicted_phase'), exist_ok=True)
            phase_path = os.path.join(config['results_dir'], 'predicted_phase', fname)
            np.savetxt(phase_path, np.column_stack((pred_ph0, pred_ph1)), delimiter=' ', fmt='%.6f')