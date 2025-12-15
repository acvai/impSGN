# impSGN

**Improved Semantic-Guided Network (impSGN)** for Skeleton-Based Action Recognition

This project implements the **Improved Semantic-Guided Network (impSGN)** model, based on our paper:  

> *"Improved Semantic-Guided Network for Skeleton-Based Action Recognition"*

The implementation is built on top of the **SGN repository**, with enhancements to semantic-guided feature learning, temporal modeling, and classification performance.

---

## Table of Contents

- [Background](#background)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Dataset](#dataset)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Results](#results)  
- [References](#references)  
- [License](#license)

---

## Background

Skeleton-based action recognition uses **human joint coordinates** over time to classify actions. Traditional methods rely on hand-crafted features or simple deep learning models.  

The original **SGN model** introduced **semantic-guided networks** to enhance skeleton representation learning. Our **impSGN** extends SGN by:

- Improving semantic-guided module design  
- Enhancing temporal modeling  
- Achieving higher recognition accuracy on benchmark datasets  

---

## Features

- Fully implemented **impSGN network**  
- Based on PyTorch  
- Training and evaluation scripts included  
- Compatible with standard skeleton datasets  

---

## Installation

### 1. Clone this repository
```bash
git clone git@github.com:acvai/impSGN.git
cd impSGN
```

### 2. Create a conda environment
```bash
conda create -n impSGN python=3.11
conda activate impSGN
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> Make sure PyTorch is installed for your CUDA version if using GPU.

---

## Usage

### Training
```bash
python train.py --dataset NTU --model impSGN --epochs 50
```

### Evaluation
```bash
python test.py --dataset NTU --model impSGN --weights path/to/checkpoint.pth
```
Adjust dataset paths and hyperparameters in `config.yaml` or command-line arguments.

---

## Dataset

We recommend using **NTU RGB+D** or similar skeleton datasets. Preprocess the data into **joint coordinates sequences**.  

---

## Training & Evaluation

- Use provided scripts to train the model  
- Save checkpoints in `checkpoints/`  
- Evaluate using the provided evaluation scripts  

---

## Results

| Dataset | Accuracy  (SGN)   | Accuracy    (impSGN)      |
|---------|----------------   |------------------         |
| NTU60  | 89.0%       CS     | 89.7%      CS           |
| NTU120 | 79.2%       CSub   | 84.6%        CSub         |
> Numbers are based on our experiments (replace with your actual results).

---

## References

1. Original SGN repository: [https://github.com/your-reference/SGN](https://github.com/your-reference/SGN)  
2. Paper: *"Improved Semantic-Guided Network for Skeleton-Based Action Recognition"*  

---
@article{MANSOURI2024104281,
title = {Improved semantic-guided network for skeleton-based action recognition},
journal = {Journal of Visual Communication and Image Representation},
volume = {104},
pages = {104281},
year = {2024},
issn = {1047-3203},
doi = {https://doi.org/10.1016/j.jvcir.2024.104281},
url = {https://www.sciencedirect.com/science/article/pii/S1047320324002372},
author = {Amine Mansouri and Toufik Bakir and Abdellah Elzaar},
keywords = {Deep learning, Human Action Recognition (HAR), Convolutional Neural Networks (CNNs), Graph Convolutional Networks (GCNs), Attention mechanism},
abstract = {A fundamental issue in skeleton-based action recognition is the extraction of useful features from skeleton joints. Unfortunately, the current state-of-the-art models for this task have a tendency to be overly complex and parameterized, which results in low model training and inference time efficiency for large-scale datasets. In this work, we develop a simple but yet an efficient baseline for skeleton-based Human Action Recognition (HAR). The architecture is based on adaptive GCNs (Graph Convolutional Networks) to capture the complex interconnections within skeletal structures automatically without the need of a predefined topology. The GCNs are followed and empowered with an attention mechanism to learn more informative representations. This paper reports interesting accuracy on a large-scale dataset NTU-RGB+D 60, 89.7% and 95.0% on respectively Cross-Subject, and Cross-View benchmarks. On NTU-RGB+D 120, 84.6% and 85.8% over Cross-Subject and Cross-Setup settings, respectively. This work provides an improvement of the existing model SGN (Semantic-Guided Neural Networks) when extracting more discriminant spatial and temporal features.}
}


## License

This project is licensed under the MIT License. See `LICENSE` for details.

