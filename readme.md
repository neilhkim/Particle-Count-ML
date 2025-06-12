# Particle Count ML

Neural network models for counting particles in microscopy images using PyTorch. The performance of this model is to be compared with my other project - "Multiple hypothesis test based particle counting".

## Features

- Multiple CNN architectures (Baseline, Enhanced, MobileNet+CORAL)
- CORAL ordinal regression for ordered counting
- Focal Loss for class imbalance handling
- Automated model comparison and classification reports

## Quick Start

```bash
# Install dependencies
pip install torch torchvision timm coral-pytorch scikit-learn matplotlib tqdm

# Train multiple architectures on synthetic data
python train_models.py --synthetic 10000 --arch baseline,enhanced,mobilenet

# Train on real dataset
python train_models.py --data_root trainset05 --arch mobilenet --epochs 80

# Resume training from checkpoint - to be implemented
```

## Model Architectures

1. **Baseline CNN**: Simple 3-layer network for quick experiments (23.8K parameters)
2. **Enhanced CNN**: Deeper network with LeakyReLU and batch normalization (422.4K parameters)
3. **MobileNet+CORAL**: Pretrained backbone with ordinal regression head (1.6M parameters)

## Scripts

- `train_models.py`: Multi-architecture training with comparison plots
- `generate_dataset.py`: Synthetic particle dataset generation with PSF convolution
- `predict_test_dataset_labels.py`: Predict particle counts on test images with confidence scores

## Performance

Typical results on 5-class particle counting (0-4 particles):
- Baseline CNN: to be tested thoroughly
- Enhanced CNN: to be tested thoroughly
- MobileNet+CORAL: to be tested thoroughly

## Usage Examples

```bash
# Compare all architectures
python train_models.py --arch baseline,enhanced,mobilenet --epochs 50
```

## CORAL Ordinal Regression

Instead of independent classes, CORAL treats counting as ordered decisions:
- Traditional: P(0), P(1), P(2), P(3), P(4)
- CORAL: P(≥1), P(≥2), P(≥3), P(≥4)

This approach better handles the natural ordering of particle counts.

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision, timm, coral-pytorch, scikit-learn, matplotlib, tqdm

## Project Structure

```
Particle-Count-ML/
├── train_models.py               # Multi-architecture comparison
├── generate_dataset.py           # Synthetic dataset generation
├── predict_test_dataset_labels.py # Test image prediction
├── README.md
├── requirements.txt
└── .gitignore
```

Note: Model checkpoints (*.pth), datasets (trainset*/), and results (*.png, *.csv) are excluded from version control.