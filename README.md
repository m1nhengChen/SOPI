## News
* **2024/5/22** Domain randomization for bridging the domain gap between DRRs and real X-rays is now available at [link](https://github.com/m1nhengChen/cdreg/blob/main/domain_randomization.py)
* **2023/12/15** Our paper [M Chen, Z Zhang, S Gu, Y Kong, Embedded Feature Similarity Optimization with Specific Parameter Initialization for 2D/3D Medical Image Registration](https://arxiv.org/abs/2305.06252)  has been accepted to [ICASSP 2024](https://2024.ieeeicassp.org/), Seoul, Korea.  :tada:
## Introduction

<img src="/img/framework.png" width="500px"/>

In this work, we propose a novel two-stage 2D/3D registration framework,
Embedded Feature Similarity Optimization with Specific Parameter Initialization (SOPI), which can align the images automatically without a large amount
of real X-ray data for training and weaken the effect of incorrect initialization on
the registration algorithm. In this framework, we propose a regressive parameterspecific module, Rigid Transformation Parameter Initialization (RTPI) module,
to initialize pose parameter and an iterative fine-registration network to align the
two images precisely by using embedded features. The framework estimates the
transformation parameter that best aligns two images using one intra-operative
x-ray and one pre-operative CT as input.

<img src="/img/RTPI.png" width="700px"/>

<img src="/img/composite_encoder.png" width="800px"/>

## Setup

### Prerequisites
- Linux
- NVIDIA GPU + CUDA
- python 3.7 (recommended)

### Getting Started
- Install torch, torchvision from https://pytorch.org/. It has been tested with torch 1.9.1.
- Check requirements.txt for dependencies. You can use pip install:
```bash
pip install -r requirements.txt
```
### Train RTPI
- We strongly recommand you to use the RTPI_v3 model.
- The average running time list blow are the test results on the RTX 3090.

  |Version | RTPI-V1 | RTPI-V2 | RTPI-V3 |
  |--------- |---------|---------|---------|
  |Avg.time |0.15s | 0.069s  | 0.066s  |

```bash
cd ./src
python train_RTPI.py
```
### Train composite encoders(fine registration module)
```bash
cd ./src
python train_composite_encoder.py
```
## Citation
If you use this code for your research, please cite our paper:
```
@inproceedings{chen2024embedded,
  title={Embedded Feature Similarity Optimization with Specific Parameter Initialization for 2D/3D Medical Image Registration},
  author={Chen, Minheng and Zhang, Zhirun and Gu, Shuheng and Kong, Youyong},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1521--1525},
  year={2024},
  organization={IEEE}
}
```
**(PS:The CPU-Net file contains some early immature ideas, which have limited reference value.)**
## Acknowledgements

Special thanks to the students and professors in Jiangsu Provincial Joint International Research
Laboratory of Medical Information Processing, Southeast University, Nanjing, China, who provided assistance, inspiration and support for our work.

## FAQ

List possible frequently asked questions and their answers.

If you have any other questions, feel free to contact us.

## Contact

If you need to get in touch with us, you can reach us through the following channels:

- Email: mh_chen@seu.edu.cn
