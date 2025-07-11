# Deep Autoencoder with SVD Convergence

This repository contains implementations of learnable weighted dimensionality reduction frameworks for various fluid dynamics and turbulence datasets using deep autoencoders with SVD convergence.

## Overview

The project implements hybrid autoencoder architectures that combine deep learning with singular value decomposition (SVD) for dimensionality reduction and surrogate modeling across multiple fluid dynamics problems.

## Datasets

### 1D Kuramoto-Sivashinsky (KS) Dimensionality Reduction
- **Data Files**: `ks_512_extended.mat`, `ks_1024_extended.mat`, `ks_2048_extended.mat`
- **Description**: Extended Kuramoto-Sivashinsky equation data for dimensionality reduction analysis

### 3D Homogeneous Isotropic Turbulence (HIT)
- **Data Source**: Johns Hopkins University (JHU) Turbulence Database
- **Configuration**: See manuscript for specific configuration details
- **Download**: Available from JHU turbulence database based on manuscript specifications

### 1D Wave Koopman
- **Data Generation**: Generated within the training file
- **Description**: Synthetic data for Koopman operator analysis of traveling waves

### 2D Cylinder Koopman
- **Data File**: `VORTALL.mat` 
- **Description**: Vorticity data for flow over cylinder analysis

### 1D Viscous Burgers' Surrogate
- **Data Generation**: Part of training script (uploaded as ESM)
- **Description**: Synthetic data for surrogate modeling using LSTM

### 2D Shallow Water
- **Data Source**: Generated using codes from [CAE_LSTM_ROMS](https://github.com/Romit-Maulik/CAE_LSTM_ROMS.git)
- **Repository**: https://github.com/Romit-Maulik/CAE_LSTM_ROMS.git

### 3D Viscous Burgers' Surrogate
- **Data Source**: Generated using [Apebench](https://github.com/tum-pbs/apebench.git)
- **Configuration**: See manuscript for specific configuration details
- **Repository**: https://github.com/tum-pbs/apebench.git

## Code Files

### Dimensionality Reduction Frameworks

#### `hybrid_ae_ks.py`
- **Description**: Learnable weighted dimensionality reduction framework for 1D KS data
- **Features**: Hybrid autoencoder with SVD convergence for Kuramoto-Sivashinsky equation

#### `hybrid_ae_3d.py`
- **Description**: Learnable weighted dimensionality reduction framework for 3D Turbulence data
- **Features**: 3D autoencoder implementation for homogeneous isotropic turbulence

### Surrogate Modeling with LSTM

#### `lstm_training_1d_vb.py`
- **Description**: Surrogate modeling using LSTM with learnable weighted dimensionality reduction for 1D Viscous Burgers' data
- **Features**: LSTM-based surrogate model with autoencoder preprocessing

#### `lstm_training_shallow_water.py`
- **Description**: Surrogate modeling with learnable weighted dimensionality reduction for 2D Shallow Water data
- **Features**: 2D LSTM surrogate model for shallow water equations

#### `lstm_training_3d_vb.py`
- **Description**: Surrogate modeling with learnable weighted dimensionality reduction for 3D Viscous Burgers'
- **Features**: 3D LSTM surrogate model for viscous Burgers' equation

### Data Generation

#### `generate_data.py`
- **Description**: Generate 1D Viscous Burgers' data for surrogate modeling using LSTM
- **Features**: Synthetic data generation for training and testing

### Koopman Forecasting

#### `wave.py`
- **Description**: Koopman forecast with learnable weighted dimensionality reduction for 1D traveling wave
- **Features**: Koopman operator analysis for wave dynamics

#### `cylinder_vortall.py`
- **Description**: Koopman forecast with learnable weighted dimensionality reduction for 2D flow over cylinder
- **Features**: Koopman operator analysis for cylinder wake dynamics

## Installation and Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd deep-ae-with-svd-convergence
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download or generate the required datasets as specified above.

## Usage

### Running Dimensionality Reduction
```bash
# For 1D KS data
python hybrid_ae_ks.py

# For 3D turbulence data
python hybrid_ae_3d.py
```

### Running Surrogate Models
```bash
# For 1D Viscous Burgers'
python lstm_training_1d_vb.py

# For 2D Shallow Water
python lstm_training_shallow_water.py

# For 3D Viscous Burgers'
python lstm_training_3d_vb.py
```

### Running Koopman Forecasting
```bash
# For 1D traveling wave
python wave.py

# For 2D cylinder flow
python cylinder_vortall.py
```

### Data Generation
```bash
# Generate 1D Viscous Burgers' data
python generate_data.py
```

## Data Sources

- **JHU Turbulence Database**: For 3D HIT data
- **CAE_LSTM_ROMS Repository**: For 2D Shallow Water data generation
- **Apebench Repository**: For 3D Viscous Burgers' data generation

## Citation

If you use this code in your research, please cite the associated manuscript.

@misc{somasekharan2025kolmogorovbarrierlearnableweighted,
      title={Beyond the Kolmogorov Barrier: A Learnable Weighted Hybrid Autoencoder for Model Order Reduction}, 
      author={Nithin Somasekharan and Shaowu Pan},
      year={2025},
      eprint={2410.18148},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.18148}, 
}