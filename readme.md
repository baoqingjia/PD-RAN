# how to run the code?

## run python script

all you need is `resnet_attention.ipynb`

## about the data
```bash
(base) ➜  data git:(main) ✗ tree
.
├── X.npy
├── Y.npy
└── Z.npy

0 directories, 3 files
```

where `X.npy` is the training data of shape `(batch_size, spectrum_len, value)`, `value` contains the real part and the image part, it is a degree range from 0 to 360, but there may exists some values below 0 and exceed 360. 

`Y.npy` is the expected value of the spectrum real part

`Z.npy` is the label, which is correspond to the `value` which is in two values , namely `(ph0, ph1)`, these two are float degrees

## citation about the network

```txt
@inproceedings{wang2017residual,
  title={Residual attention network for image classification},
  author={Wang, Fei and Jiang, Mengqing and Qian, Chen and Yang, Shuo and Li, Cheng and Zhang, Honggang and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3156--3164},
  year={2017}
}
```