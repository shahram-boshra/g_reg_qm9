# Graph Neural Network for QM9 Property Prediction

This repository contains code for training and evaluating a Graph Neural Network (GNN) model to predict molecular properties from the QM9 dataset. The model leverages PyTorch Geometric for graph data handling and supports a flexible architecture defined by the number and types of GNN layers.

## Table of Contents

* [Description](#description)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Dependencies](#dependencies)
* [Configuration](#configuration)
* [Usage](#usage)
* [Code Overview](#code-overview)
* [Model Details](#model-details)
* [Evaluation](#evaluation)
* [Plots](#plots)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Description

This project implements a GNN model to predict molecular properties from the QM9 dataset. It provides a flexible framework for experimenting with different GNN architectures and training parameters. The code includes data loading, model definition, training, evaluation, and visualization tools.

## Repository Structure

.
├── config.yaml         # Configuration file for training parameters and model architecture
├── dataset.py          # QM9 dataset loading and preprocessing
├── device_utils.py     # Device management utilities (CPU/GPU)
├── exceptions.py       # Custom exception classes
├── models.py           # GNN model definition with configurable layers
├── README.md           # This file
├── training_utils.py   # Training, validation, and plotting utilities
└── main.py             # Main script for training and evaluation


## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/shahram-boshra/g_reg_qm9.git (or git@github.com:shahram-boshra/g_reg_qm9.git)
    cd g_reg_qm9
    ```

2.  **Prepare the QM9 Dataset:** Ensure the QM9 dataset is downloaded and placed in the directory specified by `root_dir` in `config.yaml`.

## Dependencies

* `torch`
* `torch-geometric`
* `scikit-learn`
* `numpy`
* `matplotlib`
* `pyyaml`
* `pydantic`

Install the required packages using pip:

```bash
pip install torch torch-geometric scikit-learn numpy matplotlib pyyaml pydantic
Configuration
The training and model architecture are configured through config.yaml. Key parameters include:

Dataset:

root_dir: Path to the QM9 dataset directory.
target_indices: List of indices corresponding to the target properties.
subset_size: Number of samples to use from the dataset.
train_split, valid_split: Proportions for train, validation, and test splits.
Model:

num_layers: The number of graph convolutional layers in the GNN.
layer_types: A list specifying the type of each GNN layer in sequence (e.g., ["gcn", "gat", "transformer_conv"]). Supported types include gcn, gat, sage, gin, graph_conv, transformer_conv, and custom_mp. The length of this list should typically match num_layers.
hidden_channels: Number of hidden units in the GNN layers.
dropout_rate: Dropout rate.
gat_heads: Number of attention heads for GATConv layers.
transformer_heads: Number of attention heads for TransformerConv layers.
Training:

batch_size: Training batch size.
learning_rate: Learning rate for the optimizer.
weight_decay: Weight decay for the optimizer.
step_size, gamma: Parameters for StepLR learning rate scheduler.
reduce_lr_factor, reduce_lr_patience: Parameters for ReduceLROnPlateau learning rate scheduler.
early_stopping_patience, early_stopping_delta: Parameters for early stopping.
l1_regularization_lambda: L1 regularization strength.
Example config.yaml:

YAML

data:
  root_dir: 'C:/Chem_Data/qm9'
  target_indices: [3, 6, 12]
  subset_size: 10000
  train_split: 0.8
  valid_split: 0.1

model:
  num_layers: 3
  layer_types: ["gat", "gcn", "transformer_conv"]
  hidden_channels: 128
  dropout_rate: 0.5
  gat_heads: 2
  transformer_heads: 3
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  step_size: 50
  gamma: 0.5
  reduce_lr_factor: 0.5
  reduce_lr_patience: 10
  early_stopping_patience: 50
  early_stopping_delta: 0.0001
  l1_regularization_lambda: 0.001

Usage
Run the Main Script: Execute main.py to start the training and evaluation process:

Bash

python main.py
The script will:
Load the configuration from config.yaml.
Load and preprocess the QM9 dataset.
Split the dataset into training, validation, and test sets.
Initialize the GNN model based on the num_layers and layer_types specified in the configuration.
Train the model, applying early stopping and learning rate scheduling.
Evaluate the model on the test set.
Save test set predictions and targets to .npy files.
Generate plots of training/validation losses and evaluation metrics.
Code Overview
dataset.py: Loads and preprocesses the QM9 dataset, creating a PyTorch Geometric Dataset.
models.py: Defines the MGModel GNN architecture. The model's layers are dynamically created based on the num_layers and layer_types specified in the configuration. It supports various GNN layer types and includes a custom message passing layer (CustomMPLayer). The forward method handles both torch_geometric.data.Data objects and separate tensors for flexibility.
training_utils.py: Implements training and evaluation logic, including:
EarlyStopping: For preventing overfitting.
Trainer: Manages the training loop and evaluation.
Plot: Generates plots of training losses and evaluation metrics.
calculate_metrics: Calculates evaluation metrics.
main.py: Main script to execute the training and evaluation pipeline.
config_loader.py: Loads and validates the configuration from config.yaml using pydantic.
device_utils.py: Handles device selection (CPU or GPU).
exceptions.py: Defines custom exceptions for error handling.
Model Details
The MGModel's architecture is now defined by the num_layers and the layer_types list in the config.yaml. It supports the following GNN layers:

GCNConv
GATConv
SAGEConv
GINConv
GraphConv
TransformerConv
CustomMPLayer (a custom message passing layer)
The model includes:

Batch normalization.
Dropout.
L1 regularization.
Global mean pooling.
Evaluation
The model's performance is evaluated using:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R2 score
Explained variance
Plots
The script generates plots of:

Training and validation losses vs. epochs.
MAE, MSE, R2, and explained variance vs. epochs.
Contributing
Contributions are welcome! Please submit pull requests or open issues for any improvements or bug fixes.

License
This project is licensed under the MIT License - see the LICENSE file for details.   


Acknowledgments
This project utilizes the QM9 dataset.
We acknowledge the developers of PyTorch and PyTorch Geometric for providing powerful tools for graph neural networks.
We thank the open-source community for their contributions to the dependencies used in this project.