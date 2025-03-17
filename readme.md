## Requirement
This project is based on Python 3.6.0 and Pytorch 1.8.0

## data
Due to upload limitations, we only provide some simulated and in vivo data for training and testing (each containing 100 spectra). Please refer to "data/".

Each h5 file contains data_rm (spectrum without phase correction, including real and imaginary parts), gt_real (manually phase-corrected spectrum), gt_phase (manually phase-corrected phase)

## Training
python train.py

The 'data_dir' of config is the data path, and 'save_dir' is the pre-trained model save path

## Pre-trained models
Models trained based on simulation and in vivo data are available at 'checkpoint/'

## Testing
python test.py

The 'data_dir' of config is the data path, 'save_dir' is the pre-trained model loading path, and 'results_dir' is the test results save path.

## References
[1] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

[2] Wang F, Jiang M, Qian C, et al. Residual attention network for image classification[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 3156-3164.