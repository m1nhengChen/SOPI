## Introduction

<img src="/img/framework.png" width="600px"/>

In this [paper](https://arxiv.org/abs/2305.06252), we propose a novel two-stage 2D/3D registration framework,
Embedded Feature Similarity Optimization with Specific Parameter Initialization (SOPI), which can align the images automatically without a large amount
of real X-ray data for training and weaken the effect of incorrect initialization on
the registration algorithm. In this framework, we propose a regressive parameterspecific module, Rigid Transformation Parameter Initialization (RTPI) module,
to initialize pose parameter and an iterative fine-registration network to align the
two images precisely by using embedded features. The framework estimates the
transformation parameter that best aligns two images using one intra-operative
x-ray and one pre-operative CT as input.
<img src="/img/RTPI.png" width="800px"/>
<img src="/img/composite_encoder.png" width="900px"/>
## Setup

### Prerequisites
- Linux
- NVIDIA GPU + CUDA
- python 3.7 (recommended)

### Getting Started
- Install torch, torchvision from https://pytorch.org/. It has been tested with torch 1.13.0 and torchvision 0.8.1.
- Check requirements.txt for dependencies. You can use pip install:
```bash
pip install -r requirements.txt
```
## Citation
If you use this code for your research, please cite our paper:
```
@article{chen2023embedded,
  title={Embedded Feature Similarity Optimization with Specific Parameter Initialization for 2D/3D Registration},
  author={Chen, Minheng and Zhang, Zhirun and Gu, Shuheng and Kong, Youyong},
  journal={arXiv e-prints},
  pages={arXiv--2305},
  year={2023}
}
```
**(PS: The CPU-Net file contains some early immature ideas, it does not have much value for reference)**
