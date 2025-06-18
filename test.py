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


current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

config = {
    'model_type': 'PDRAN',
    'current_time': current_time,
    'batch_size': 4,
    'cuda_device': torch.device("cuda:0"),
    'data_dir': 'data/vivo/test/', # 'vivo' or 'simu'
    'save_dir': 'checkpoint/vivo/best.pth', # 'vivo' or 'simu'
    'results_dir' : 'results/vivo/', # 'vivo' or 'simu'
}

device = config['cuda_device']
log_dir = Path("log/test")
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f"{config['current_time']}_{config['model_type']}_test_log.txt"

test_dataset = TxtDataset(root_dir=config['data_dir'])

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
        for batch_idx, (sepc, phase, fnames) in enumerate(test_loader):
            sepc, phase = sepc.to(device), phase.to(device)

            # Forward pass
            output_phase = model(sepc)
            loss = criterion(output_phase, phase).item()
            test_loss += loss

            # Print and log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                log_message = f"Step [{batch_idx + 1}/{len(test_loader)}], test_Loss: {loss:.4f}"
                print(log_message)
                log_file.write(log_message + "\n")
                log_file.flush()

            # Convert tensors to NumPy
            output_phase = output_phase.cpu().detach().numpy()
            sepc = sepc.cpu().detach().numpy()
            phase = phase.cpu().detach().numpy()

            # Process each sample in batch
            for i in range(sepc.shape[0]):
                out_ph0, out_ph1 = output_phase[i, :2]
                gt_ph0, gt_ph1 = phase[i, :2]

                N = 64 * 1024
                out_num = np.linspace(0, 1, N)
                out_phase = (out_ph0 + out_ph1 * out_num) * np.pi / 180
                gt_phase = (gt_ph0 + gt_ph1 * out_num) * np.pi / 180

                sepc_real, sepc_imag = sepc[i, 0].flatten(), sepc[i, 1].flatten()

                out_real = sepc_real * np.cos(out_phase) - sepc_imag * np.sin(out_phase)
                out_imag = sepc_real * np.sin(out_phase) + sepc_imag * np.cos(out_phase)

                gt_real = sepc_real * np.cos(gt_phase) - sepc_imag * np.sin(gt_phase)

                # Get the corresponding filename
                fname = fnames[i]

                # Save real part comparison plot
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
    final_log = f"Average Test Loss: {avg_test_loss:.4f}"
    print(final_log)
    log_file.write(final_log + "\n")