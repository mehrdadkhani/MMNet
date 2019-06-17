# MMNet
MMNet is a massive MIMO signal detection scheme based on light online learning with neural networks that extends to correlated channel scenarios. 

## Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Repository structure](#repository-structure)

## Introduction
This repository contains MMNet signal detection model, the channels dataset, and benchmarking detection schemes discusssed in the paper "Adaptive Neural Signal Detection for Massive MIMO" (https://arxiv.org/abs/1906.04610). On i.i.d. Gaussian channels, MMNet requires two orders of magnitude fewer operations than existing deep learning schemes but achieves near-optimal performance. On spatially-correlated channels, it achieves the same error rate as the next-best learning scheme (OAMPNet) at 2.5dB lower SNR and with at least 10x less computational complexity. MMNet is also 4--8dB better overall than a classic linear scheme like the minimum mean square error (MMSE) detector.

## Citation
You may cite this project as:
```
@article{khani2019adaptive,
  title={Adaptive Neural Signal Detection for Massive MIMO},
  author={Khani, Mehrdad and Alizadeh, Mohammad and Hoydis, Jakob and Fleming, Phil},
  journal={arXiv preprint arXiv:1906.04610},
  year={2019}
}
```

## Repository structure
Find MMNet and other learning based schemes in ``./learning_based`` directory. Minimum mean square error (MMSE), Approximated message passaing (AMP), Semidefinite relaxation (SDR), Multistage interference cancelation (BLAST), and Maximum-likelihood optimal (ML) are located under ``./classic``. In order to reproduce the simulated correlated channels using 3D-3GPP model please refer to ``./channels`` directory.  
