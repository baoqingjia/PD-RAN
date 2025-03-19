## Requirement

This project is based on Python 3.6.0 and Pytorch 1.8.0

## data

Due to upload limitations, we only provide some simulated and in vivo data for training and testing (each containing 100 spectra). Please refer to `data/`.

Each h5 file contains data_rm (spectrum without phase correction, including real and imaginary parts), gt_real (manually phase-corrected spectrum), gt_phase (manually phase-corrected phase)

## Training

```bash
python train.py
```

The `data_dir` of config is the data path, and 'save_dir' is the pre-trained model save path

## Pre-trained models

Models trained based on simulation and in vivo data are available at `checkpoint/`

## Testing

python test.py

The `data_dir` of config is the data path, 'save_dir' is the pre-trained model loading path, and 'results_dir' is the test results save path.

## References

```txt
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
