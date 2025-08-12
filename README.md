# Bilevel Additive Taylor Model (BATM)

This repository contains the official implementation for the paper: "Interpretable Bilevel Additive Taylor Model for Datasets with Noisy Labels and Imbalanced Classes".

BATM is an interpretable and robust framework that integrates bilevel optimization with sparse neural additive modeling. It is designed to handle real-world data challenges such as label noise, class imbalance, and outliers, while maintaining high interpretability and predictive performance.

## Model Architecture

The core of BATM consists of a two-level optimization process:

- **Lower-level Model**: A sparse `BATM_TaylorNetwork` that learns feature representations and makes predictions. It uses concept encoders to group features and a Tucker-decomposed Taylor expansion for high-order interactions.
- **Upper-level Model**: A `MetaWeightNet` that acts as a meta-learner. It dynamically learns to assign weights to training samples, effectively down-weighting noisy or less informative samples and up-weighting those from minority classes.

This bilevel structure allows BATM to simultaneously optimize for robustness and accuracy.

## Project Structure

```
BATM_Project/
├── data/
│   └── dataset_loader.py       # Data loaders for synthetic and real-world datasets.
├── logs/
│   └── ...                     # Directory for logs and model checkpoints.
├── models/
│   ├── batm_concept_encoder.py # The BATM concept encoder module.
│   └── batm_taylor_network.py  # The main BATM Taylor Network architecture.
├── optimizer/
│   └── bilevel_optimizer.py    # Bilevel optimization logic, including gradient updates.
├── configs/
│   └── default_config.yaml     # Configuration file for hyperparameters.
├── main.py                     # Main script to run experiments.
├── README.md                   # This file.
├── requirements.txt            # Required packages.
└── utils/                      # Utility functions package.
    ├── __init__.py
    ├── data_utils.py           # Data processing utilities.
    └── training_utils.py       # Training utilities (metrics, early stopping, etc.).
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/BATM_Project.git
   cd BATM_Project
   ```
2. Install the required dependencies. It is recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Experiments can be configured and run through the main script. Hyperparameters are managed via configuration files.

```bash
# Run an experiment using the default configuration
python main.py --config configs/default_config.yaml
```

You can create new `.yaml` files in the `configs/` directory to define your own experiments.

## Citation

If you find this work useful in your research, please consider citing our paper:

```
@article{zhou2025interpretable,
  title={Interpretable Bilevel Additive Taylor Model for Datasets with Noisy Labels and Imbalanced Classes},
  author={Zhou, Wenxing and Xu, Chao and Peng, Lian and Zhang, Xuelin},
  journal={Working Paper},
  year={2025}
}
```
