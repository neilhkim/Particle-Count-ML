# Particle Count ML

Neural network models for counting particles in microscopy images using PyTorch. The performance of this model is to be compared with my other project - "Multiple hypothesis test based particle counting".

## Features

- Multiple CNN architectures (Baseline, Enhanced, MobileNet+CORAL)
- CORAL ordinal regression for ordered counting tasks
- Focal Loss for handling class imbalance
- Automated model comparison with training history plots
- Per-model random seed control for reproducible experiments
- Adaptive learning rate scheduling based on validation performance
- Synthetic dataset generation with realistic PSF simulation

## Quick Start

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

1. **Baseline CNN**: Simple 3-layer network for quick experiments (~24K parameters)
2. **Enhanced CNN**: Deeper network with LeakyReLU and batch normalization (~422K parameters)  
3. **MobileNet+CORAL**: Pretrained MobileNetV3-Small backbone with ordinal regression head (~1.6M parameters)

## Scripts

- `train_models.py`: Multi-architecture training with automatic comparison plots
- `generate_dataset.py`: Synthetic particle dataset generation with Gaussian PSF convolution
- `predict_test_dataset_labels.py`: Batch prediction on test images with confidence scores

## Training Features

### Adaptive Learning Rate
- Uses `ReduceLROnPlateau` scheduler that reduces LR when validation loss plateaus
- More stable than epoch-based scheduling and compatible with resume functionality
- Patience of 5 epochs before reducing LR by factor of 0.5

### Per-Model Random Seeds
```bash
# Different seed for each architecture
python train_models.py --arch enhanced,mobilenet --seed 42 123

# Same seed for all models (original behavior)  
python train_models.py --arch enhanced,mobilenet --seed 42
```

### Data Augmentation
- Random horizontal/vertical flips and rotations (particles are rotationally symmetric)
- Random affine transformations (translation, scaling)
- Optional MixUp regularization with `--mixup` parameter

## Performance

Performance on 5-class particle counting (0-4 particles): **To be tested thoroughly**

- **Baseline CNN**: Performance testing in progress
- **Enhanced CNN**: Performance testing in progress  
- **MobileNet+CORAL**: Performance testing in progress

*Results will be updated once comprehensive benchmarking is completed*

![2025-06-11 loss and accuracy](https://github.com/user-attachments/assets/10fb20b3-b09a-4e1d-9fe2-c86b3ccbec8a)

## CORAL Ordinal Regression

Instead of treating particle counts as independent classes, CORAL uses the natural ordering:
- **Traditional**: P(0), P(1), P(2), P(3), P(4) - treats each count independently
- **CORAL**: P(≥1), P(≥2), P(≥3), P(≥4) - learns cumulative probabilities

This approach better captures that 2 particles is "closer" to 3 particles than to 0 particles.

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

- Python 3.7+
- PyTorch 1.9+
- torchvision, timm, coral-pytorch, scikit-learn, matplotlib, tqdm, numpy

## Project Structure

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

Trained models are saved with descriptive names including parameter count:
- `enhanced-CNN_422.3Kparams-trainset05-epoch15.pth`
- `mobilenet-CORAL_1.6Mparams-synthetic-epoch25.pth`

Each checkpoint includes model state, optimizer state, training metrics, and metadata for easy resuming.

## Troubleshooting

**Low accuracy (~20%)**:
- Check learning rate isn't too low due to aggressive scheduling
- Verify data loading (should see different accuracies with different seeds)
- Ensure sufficient training epochs (try 20+ for real datasets)

**Memory issues**:
- Reduce batch size with `--batch 32`
- Train one architecture at a time
