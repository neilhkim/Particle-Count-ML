# Particle Count ML

This project focuses on developing neural network models for counting particles in microscopy images using PyTorch. The performance of these models will be compared with another project that uses a different method called "Multiple hypothesis test based particle counting."

## Features

- Multiple CNN architectures (Baseline, Enhanced, MobileNet+CORAL)
- CORAL ordinal regression for ordered counting tasks
- Focal Loss for handling class imbalance
- Automated model comparison with training history plots
- Per-model random seed control for reproducibility.
- Adaptive learning rate scheduling based on validation performance.
- Synthetic dataset generation with realistic Point Spread Function (PSF) simulation.

## ## Getting Started

Installation and usage commands:

```bash
# Install dependencies
pip install torch torchvision timm coral-pytorch scikit-learn matplotlib tqdm

# Train on real dataset with different seeds per model
python train_models.py --data_root trainset05 --arch enhanced,mobilenet --epochs 20 --seed 42 123

# Train single model with custom learning rate
python train_models.py --data_root trainset05 --arch enhanced --epochs 15 --lr 0.001

# Resume training from checkpoint (coming soon)
python train_models.py --resume enhanced-CNN_422.3Kparams-trainset05-epoch10.pth
```

## Model Architectures

The models included in this project are as follows:

1. **Baseline CNN**: A simple 3-layer network designed for quick experiments, with approximately 24K parameters.
2. **Enhanced CNN**: A deeper network featuring LeakyReLU and batch normalization, totaling about 422K parameters.
3. **MobileNet plus CORAL**: A pretrained MobileNetV3-Small backbone combined with an ordinal regression head, amounting to approximately 1.6M parameters.

## Scripts Overview

- `train_models.py`: The main script for training multiple architectures and generating comparison plots.
- `generate_dataset.py`: This script generates synthetic particle datasets using Gaussian PSF convolution.
- `predict_test_dataset_labels.py`: This script performs batch predictions on test images and provides confidence scores.

## Training Features

### Adaptive Learning Rate

The project uses the `ReduceLROnPlateau` scheduler. If the validation loss plateaus, the learning rate is reduced to enhance training stability. A patience of 5 epochs is implemented to avoid premature adjustments.

### Random Seed Control

```bash
# Use different seeds for multiple architectures
python train_models.py --arch enhanced,mobilenet --seed 42 123

# Use the same seed for all models
python train_models.py --arch enhanced,mobilenet --seed 42
```

### Data Augmentation

Data augmentation techniques are incorporated, including random flips, rotations, and affine transformations. You can also enable MixUp regularization with the `--mixup` option.

## Performance

Currently, we are testing the models on a 5-class particle counting task (0 to 4 particles):

- **Baseline CNN**: Performance testing in progress
- **Enhanced CNN**: Performance testing in progress  
- **MobileNet+CORAL**: Performance testing in progress

Updates on benchmarking results will be provided soon.

![2025-06-11 loss and accuracy](https://github.com/user-attachments/assets/10fb20b3-b09a-4e1d-9fe2-c86b3ccbec8a)

## Understanding CORAL Ordinal Regression

Instead of treating particle counts as independent classes, CORAL takes an ordinal approach that recognizes the natural order of counts. Traditional models may predict counts independently (P(0), P(1), P(2), P(3), and P(4)). CORAL instead predicts cumulative probabilities (P(≥1), P(≥2), P(≥3), P(≥4)), acknowledging the closer relationship between adjacent particle counts.

## Usage Examples

```bash
# Compare all architectures with training history plots
python train_models.py --data_root trainset05 --arch baseline,enhanced,mobilenet --epochs 30

# Longer training with separate validation directory
python train_models.py --train_dir trainset05 --val_dir valset05 --arch mobilenet --epochs 80

# Enable MixUp regularization
python train_models.py --data_root trainset05 --arch enhanced --mixup 0.2 --epochs 25
```

## Requirements

To run the project, the following are needed:

- Python 3.7+
- PyTorch 1.9+
- torchvision, timm, coral-pytorch, scikit-learn, matplotlib, tqdm, numpy

## Project Structure

The project is organized as follows:

```
Particle-Count-ML/
├── train_models.py               # Main training script with multi-architecture support
├── generate_dataset.py           # Synthetic dataset generation with PSF simulation  
├── predict_test_dataset_labels.py # Batch prediction with confidence scoring
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes models, datasets, results
└── trainset*/                    # Training datasets (excluded from git)
    ├── 0/                        # Images with 0 particles
    ├── 1/                        # Images with 1 particle
    ├── 2/                        # Images with 2 particles
    ├── 3/                        # Images with 3 particles
    └── 4/                        # Images with 4 particles
```

## Model Checkpoints

Trained models are saved with names that clearly indicate their properties, including parameter count and dataset used:

- `enhanced-CNN_422.3Kparams-trainset05-epoch15.pth`
- `mobilenet-CORAL_1.6Mparams-synthetic-epoch25.pth`

## Troubleshooting Tips

If you encounter low accuracy, consider the following:

- Check if the learning rate is too low due to the aggressive scheduling.
- Verify that data loading is functioning properly; different seeds should yield varying accuracies.
- Ensure you are training for a sufficient number of epochs; 20 or more is advisable for real datasets.

If you experience memory issues, try reducing the batch size using `--batch 32`, and consider training one architecture at a time to manage resource usage.

---

If you have further questions or need assistance, please feel free to reach out.
