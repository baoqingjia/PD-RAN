import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.model import PDRAN
from utils.utils import HDF5Dataset

current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

config = {
    'model_type': 'PDRAN',
    'current_time': current_time,
    'batch_size': 4,
    'cuda_device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'data_dir': 'data/simu/test_100.h5',
    'save_dir': 'checkpoint/simu/best.pth',
    'results_dir' : 'results/simu/',
}

device = config['cuda_device']
log_dir = Path("log/test")
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f"{config['current_time']}_{config['model_type']}_test_log.txt"

test_dataset = HDF5Dataset(config['data_dir'])
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

x_list = []
y_list = []
z_list = []
outputs_list = []

with open(log_file_path, "a") as log_file:
    with torch.no_grad():
        for batch_idx, (x, y, z) in enumerate(test_loader):
            x, z = x.to(device), z.to(device)

            outputs = model(x)

            loss = criterion(outputs, z).item()
            test_loss += loss

            outputs = outputs.cpu().detach().numpy()
            x = x.cpu().detach().numpy()
            z = z.cpu().detach().numpy()

            outputs_list.append(outputs)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

            if (batch_idx + 1) % 10 == 0:
                log_message = f"Step [{batch_idx + 1}/{len(test_loader)}], test_Loss: {loss:.4f}"
                print(log_message)
                log_file.write(log_message + "\n")
                log_file.flush()

    avg_test_loss = test_loss / len(test_loader)
    final_log = f"Average Test Loss: {avg_test_loss:.4f}"
    print(final_log)
    log_file.write(final_log + "\n")

N = 64 * 1024
outputs = np.concatenate(outputs_list, axis=0)
x = np.concatenate(x_list, axis=0)
y = np.concatenate(y_list, axis=0)
z = np.concatenate(z_list, axis=0)

for k in range(outputs.shape[0]):
    out_ph0, out_ph1 = outputs[k, :2]
    gt_ph0, gt_ph1 = z[k, :2]

    out_num = np.linspace(0, 1, N)
    out_phase = (out_ph0 + out_ph1 * out_num) * np.pi / 180

    x_real, x_imag = x[k, 0].flatten(), x[k, 1].flatten()
    out_real = x_real * np.cos(out_phase) - x_imag * np.sin(out_phase)
    out_imag = x_real * np.sin(out_phase) + x_imag * np.cos(out_phase)

    gt_real = y[k, :, 0]

    results_dir = config['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(x_real, color='red')
    axes[0].set_title("x_real")
    axes[1].plot(out_real, color='red')
    axes[1].set_title("output_real")
    axes[2].plot(gt_real, color='red')
    axes[2].set_title("gt_real")
    plt.tight_layout()
    plt.savefig(f'{results_dir}k{k}.jpg')
    plt.show()
    plt.close()

    # os.makedirs('./' + results_dir + '/x_real/', exist_ok=True)
    # os.makedirs('./' + results_dir + '/out_real/', exist_ok=True)
    # os.makedirs('./' + results_dir + '/gt_real/', exist_ok=True)
    # os.makedirs('./' + results_dir + '/out_ph0_ph1/', exist_ok=True)
    # os.makedirs('./' + results_dir + '/gt_ph0_ph1/', exist_ok=True)
    #
    # np.savetxt('./' + results_dir + '/x_real/' + str(k) + '.txt', x_real, delimiter=' ')
    # np.savetxt('./' + results_dir + '/out_real/' + str(k) + '.txt', out_real, delimiter=' ')
    # np.savetxt('./' + results_dir + '/gt_real/' + str(k) + '.txt', gt_real, delimiter=' ')
    # np.savetxt('./' + results_dir + '/out_ph0_ph1/' + str(k) + '.txt', np.column_stack((out_ph0, out_ph1)), delimiter=' ')
    # np.savetxt('./' + results_dir + '/gt_ph0_ph1/' + str(k) + '.txt', np.column_stack((gt_ph0, gt_ph1)), delimiter=' ')

print("Testing finished!")