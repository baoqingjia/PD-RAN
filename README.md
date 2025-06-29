## Update on 2025/6/29

To further improve the model's generalization and performance, please train the model using data augmented with phase expansion.  
For implementation details, refer to: `scripts/pdata2txt_dataaug.m`.

## 1. Introduction

PD-RAN (Phase Model-Driven Residual Attention Network) is a deep learning-based method for automatic phase correction of Nuclear Magnetic Resonance (NMR) spectra. This project provides the implementation, training, and testing code for the PD-RAN model, as well as tools for data conversion and batch processing.

Phase correction of NMR spectra is a critical step in data processing. Traditional methods typically require manual adjustment by specialists, which is time-consuming and subjective. The PD-RAN model achieves automated and objective phase correction by learning the mapping from distorted spectra to phase parameters, making it particularly suitable for high-throughput NMR-based metabolomics research.

The project folder structure is as follows:

```
PD-RAN-main/
├── checkpoint/          # Pre-trained models for users to directly check model performance
│   ├── simu             # Models trained on simulated data
│   ├── vivo             # Models trained on real experimental data
├── data/                # Model input data directory
│   ├── simu/            # Simulated data for train and test
│   └── vivo/            # Real experimental metabolic data for train and test
│   └── samples/         # TopSpin format raw data for demonstrating format conversion 
├── log/                 # Training and testing log files
│   ├── train/           # Training logs
│   └── test/            # Testing logs
├── results/             # Model testing results
│   ├── simu/            # Simulated data test results
│   │   ├── predicted_phase/ # Predicted phase parameters
│   │   ├── corrected_real_part/       # Corrected spectral real part data
│   │   └── plot_spectra/    # Visualization results
│   └── vivo/            # Real experimental data test results
│       ├── predicted_phase/ # Predicted phase parameters
│       ├── corrected_real_part/       # Corrected spectral real part data
│       └── plot_spectra/    # Visualization results
├── scripts/             # MATLAB scripts for TopSpin data processing
│   ├── pdata2txt.m      # Convert conventional NMR formats to txt format files
│   ├── phase2pdata.m    # Apply predicted phases to raw NMR data for phase correction
├── train.py             # Training file that takes the unphased spectrum and gt_phase as input, computes the loss between the predicted and ground truth phases, and trains the neural network
├── test.py              # Testing file that evaluates model performance by computing the error between the predicted and ground truth phases, given the unphased spectrum and gt_phase as input
├── corrected.py         # Correction file that takes the unphased spectrum as input and outputs the predicted correction phase and the corrected real spectrum
├── model.py             # Model definition
├── utils.py             # Utility functions
└── requirements.txt     # Project dependencies
```

## 2. Environment Setup

This project requires the following environment setup.

### 2.1 Install Anaconda (Recommended)

We highly recommend using **Anaconda** for managing Python environments and dependencies.
Download the appropriate version for your operating system from the official website:

🔗 [Anaconda Official Download](https://www.anaconda.com/download/)

Follow the installation instructions on the website.

---

### 2.2 Create a Conda Virtual Environment

Create an isolated environment to avoid conflicts with other projects:

```bash
conda create --name <your_env_name> python=3.6
```

> 📌 This project is based on **Python 3.6.0** and **PyTorch 1.8.0**.

---

### 2.3 Dependency Installation

After cloning this repository, install the required dependencies using the following command:

```bash
conda activate <your_env_name>
pip install -r requirements.txt
```

Main dependencies include:

- python
- torch
- numpy
- matplotlib

## 3. Data Format & Conversion from Bruker Topspin

PD-RAN uses a simple text format to store NMR spectral data and phase parameters.

### 3.1 Data Format

#### 3.1.1 Spectrum File Format

Each spectrum file (`.txt`) contains one column of data:

Example (`data/vivo/train/input_spectra/301_1_1.txt`):

```
0.273562+0.153188i
0.270514+0.152569i
...
0.257358+0.149096i
```

#### 3.1.2 Phase Parameter File Format

The phase parameter file contains zero-order and first-order phase parameters in degree for all spectra (**The first column is Ph0, the second column is Ph1**):

Example (`data/vivo/train/gt_phase/301_1_1.txt`):

```
74.172285 0.030754
```

### 3.2 Data Conversion

Use the **'scripts/pdata2txt.m'** script to perform batch conversion of common NMR (Bruker) data formats to the required text format (.txt). The `data/samples/topspin_formats_data/` directory contains example TopSpin format data that you can use to practice the conversion process.

## 4. Model Usage

### 4.1 Model Training

Use the following command to train the PD-RAN model:

```bash
python train.py
```

Key parameter descriptions:

- `data_dir`: Path to your training data directory, which contains both **input spectra** files and corresponding **gt phase** files.
- `save_dir`: Directory to save the **trained model**

During training, training and validation losses will be displayed, and the best model will be automatically saved.

> **Note:** Our neural network is designed to take input data of 64k by default. For data of other sizes such as 32k, 128k, or others, you can either pad with zeros, truncate the data, or adjust the input size of the network.

### 4.2 Model Testing

Use the following command to test the trained model:

```bash
python test.py
```

In the test, the comparison between the predicted phase error and the phase-corrected result can be observed.

Key parameter descriptions:

- `--data_dir`: The **input spectra** directory stores the unphased spectra in complex form, while the **gt phase** directory contains the gt phases (used only for calculating phase correction error).
- `--save_dir`: Path to the **trained model**
- `--results_dir`: The **plot_spec** directory is used to save the results of phase correction comparisons.

### 4.3 Phase Correction

Use the following command to perform phase correction:

```bash
python corrected.py
```

In the correction process, you can obtain the predicted phase and the real part of the corrected spectrum.

Key parameter descriptions:

- `--data_dir`: The **input spectra** directory contains unphased spectra in complex form.
- `--save_dir`: Path to the **trained model**
- `--results_dir`: The **predicted_phase** and **corrected real_part** directories store the predicted phases and the phase-corrected real spectra, respectively.

  - **Predicted phase**. Predicted phase parameters are saved in the `predicted_phase` directory, in the same format as the input phase file:

  ```
  120.959229 2.542820
  ```

  - **corrected real_part**. Corrected spectra are saved as text files in the same format as the input, containing real part data.

  ```
  -0.000054
  -0.000055
  ...
  -0.000045
  ```

### 4.3 Visualization Results in TopSpin

If you want to view the corrected results in TopSpin, we also provide a MATLAB script, which is the `phase2pdata.m` file. The function of this file is to apply all predicted phase to the raw spectrum in the data folder ( `data/samples/topspin_formats_data` ) for phase correction, and write the results to the **1r (real part) and 1i (imaginary part)** files under the pdata of each spectrum.

Then, we can add the **topspin_formats_data** folder to **TopSpin** to view the corrected results of each spectrum.

## 5 Using Our Pre-trained Models

We provide models pre-trained on both **simulated** and **in vivo** data under `checkpoint/`.

## 6. References

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}

@inproceedings{wang2017residual,
  title={Residual attention network for image classification},
  author={Wang, Fei and Jiang, Mengqing and Qian, Chen and Yang, Shuo and Li, Cheng and Zhang, Honggang and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3156--3164},
  year={2017}
}
```
