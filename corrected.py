import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model import PDRAN
from utils import TxtDataset_corrected


current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

config = {
    'model_type': 'PDRAN',
    'current_time': current_time,
    'batch_size': 4,
    'cuda_device': torch.device("cuda:0"),
    'data_dir': 'data/vivo/test/',  # 'vivo' or 'simu'
    'save_dir': 'checkpoint/vivo/best.pth',  # 'vivo' or 'simu'
    'results_dir': 'results/vivo/',  # 'vivo' or 'simu'
}

device = config['cuda_device']
log_dir = Path("log/test")
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f"{config['current_time']}_{config['model_type']}_test_log.txt"

test_dataset = TxtDataset_corrected(root_dir=config['data_dir'])

test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

model = PDRAN().to(device)

if os.path.exists(config['save_dir']):
    model.load_state_dict(torch.load(config['save_dir'], map_location=device))
    print(f"Loaded model weights from {config['save_dir']}")
else:
    print("Warning: Model weights not found! Running with randomly initialized weights.")

criterion = nn.MSELoss()

model.eval()
test_loss = 0.0

sepc_list = []
phase_list = []
output_phase_list = []
fname_list = []

with open(log_file_path, "a") as log_file:
    with torch.no_grad():
        for batch_idx, (sepc, fnames) in enumerate(test_loader):
            sepc = sepc.to(device)

            # Forward pass
            output_phase = model(sepc)

            # Convert tensors to NumPy
            output_phase = output_phase.cpu().detach().numpy()
            sepc = sepc.cpu().detach().numpy()

            # Process each sample in batch
            for i in range(sepc.shape[0]):
                out_ph0, out_ph1 = output_phase[i, :2]

                N = 64 * 1024
                out_num = np.linspace(0, 1, N)
                out_phase = (out_ph0 + out_ph1 * out_num) * np.pi / 180

                sepc_real, sepc_imag = sepc[i, 0].flatten(), sepc[i, 1].flatten()
                out_real = sepc_real * np.cos(out_phase) - sepc_imag * np.sin(out_phase)
                out_imag = sepc_real * np.sin(out_phase) + sepc_imag * np.cos(out_phase)

                # Get the corresponding filename
                fname = fnames[i]

                # Create directory for real part spectra if not exists
                real_spectra_dir = os.path.join(config['results_dir'], 'corrected_real_part')
                os.makedirs(real_spectra_dir, exist_ok=True)

                # Define save path for out_real
                real_txt_path = os.path.join(real_spectra_dir, fname)
                np.savetxt(real_txt_path, out_real.reshape(-1, 1), fmt='%.6f')

                # Save corrected phase parameters
                os.makedirs(os.path.join(config['results_dir'], 'predicted_phase'), exist_ok=True)
                phase_path = os.path.join(config['results_dir'], 'predicted_phase', fname)
                np.savetxt(phase_path, np.column_stack((out_ph0, out_ph1)), delimiter=' ', fmt='%.6f')