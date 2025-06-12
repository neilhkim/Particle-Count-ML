#!/usr/bin/env python3
"""
train_particle_count_model.py  —  a messy, hand-tuned trainer + history plotting

TODO: clean up the data loading part, it's getting confusing with all the different paths
FIXME: the synthetic dataset generation could use some work
"""

# quick and dirty imports — might clean later
from __future__ import annotations
import argparse  # parsing args is always a pain
import math      # for kernel math
import os
from pathlib import Path
from typing import Tuple, Optional

import warnings
# suppress annoying lr_scheduler verbose warning - shows up constantly during training
warnings.filterwarnings("ignore", ".*verbose parameter is deprecated.*", category=UserWarning)
# Also suppress the specific torch lr_scheduler warning - had to add this after the first one didn't catch everything
warnings.filterwarnings("ignore", "The verbose parameter is deprecated", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from timm import create_model                     # got timm for mobilenet - way easier than implementing from scratch
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import CoralLoss
# from coral_pytorch.dataset import label_to_levels   # might need ordinal helper later - keeping this just in case
from tqdm import tqdm   # love those progress bars, makes me feel like something's happening
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt  # plot stuff at the end

# device setup — prefer GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True  # might speed up convs - not sure if it actually helps but why not

# utility functions - added these because I was tired of calculating param counts manually
def count_parameters(model):
    """Count trainable parameters in a model
    
    could also do sum(p.numel() for p in model.parameters()) but then I'd get frozen params too
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_param_count(count):
    """Format parameter count in human-readable form
    
    because 2340981 is harder to read than 2.3M
    """
    if count >= 1_000_000:
        return f"{count/1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count/1_000:.1f}K"
    else:
        return str(count)

def get_model_name_with_params(model_class_name, param_count):
    """Generate model name with parameter count
    
    this way I can tell models apart when I have 20 different checkpoints lying around
    """
    # Add descriptive suffixes to model names
    name_mapping = {
        'baseline': 'baseline-CNN',
        'enhanced': 'enhanced-CNN', 
        'mobilenet': 'mobilenet-CORAL'
    }
    display_name = name_mapping.get(model_class_name, model_class_name)
    param_str = format_param_count(param_count)
    return f"{display_name}_{param_str}params"

# architectures - tried a bunch of different ones, these seem to work best
class ParticleCounterCNN(nn.Module):
    def __init__(self, classes: int = 5):
        super().__init__()
        # simple 3-layer conv → global pool → linear
        # todo: maybe add another conv or more filters here - tried deeper but overfits
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 3x3 patch - could try 5x5 but probably overkill
            nn.BatchNorm2d(16),               # stabilize training - without this it was all over the place
            nn.ReLU(True),
            nn.MaxPool2d(2),                  # halves spatial dims

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),         # global average - better than flatten + linear
        )
        # note: dropout at 0.5 — could tune this, seemed like a reasonable starting point
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # rough guess, haven't really tuned this properly
            nn.Linear(64, classes)  # final logits
        )
    def forward(self, x):
        # flatten after features
        return self.classifier(self.features(x).flatten(1))
        # alt: could try x.view(x.size(0), -1) but flatten is cleaner

class EnhancedParticleCounterCNN(nn.Module):
    def __init__(self, classes: int = 5):
        super().__init__()
        # deeper net with leaky relu - tried this because regular relu was zeroing out too much
        # also made it wider because why not, we have the compute
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, True), nn.AdaptiveAvgPool2d(1),
        )
        # two-stage classifier - thought maybe it needs more capacity in the head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, True),  # small slope, 0.1 seems to work well
            nn.Linear(128, classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
        # could swap in dropout here if overfitting - haven't seen it yet though

class MobileNetCoral(nn.Module):
    """MobileNet-V3 Small backbone → 128-d embedding → CORAL layer
    
    This is for ordinal regression since particle counting is inherently ordinal
    0 < 1 < 2 < 3 < 4 particles, so we should use that structure
    CORAL paper: https://arxiv.org/abs/0704.1028 (well, the updated version)
    """
    def __init__(self, classes: int = 5):
        super().__init__()
        # use pretrained for speed, in_chans=1 to handle grayscale
        # originally tried mobilenet_v2 but v3_small is faster and similar accuracy
        self.backbone = create_model(
            'mobilenetv3_small_100', pretrained=True,
            in_chans=1, num_classes=128  # 128 dim embedding seemed reasonable
        )
        # ordinal regression: classes-1 thresholds
        # CORAL uses k-1 binary classifiers for k classes
        self.coral = CoralLayer(size_in=128, num_classes=classes-1)
        # print out to confirm shapes — comment out later
        # print(f"DEBUG: using mobilenetv3_small_100 → coral {classes-1}")
    def forward(self, x):
        return self.coral(self.backbone(x))


def coral_to_label(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORAL logits to integer count:
    sigmoid(logits) > 0.5 gives CDF per threshold, sum → count
    
    This is the key insight: CORAL outputs P(y > k) for k=0,1,2,3
    So count = sum(P(y > k) > 0.5) gives the predicted class
    """
    return (torch.sigmoid(logits) > 0.5).sum(1)

# loss functions
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        # using unreduced CE to apply focal weighting
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, inp, tgt):
        # focal loss: (1-p)^γ * log(p)
        # the idea is to down-weight easy examples and focus on hard ones
        logp = self.ce(inp, tgt)
        p = torch.exp(-logp)  # convert back to probability
        return ((1 - p)**self.gamma * logp).mean()
        # alt: use α-balancing if we have class imbalance issues

# synthetic on-the-fly dataset - because real data is hard to get
class SyntheticParticleDataset(Dataset):
    """
    Generate Poisson noisy blobs on a constant background.
    Spot kernel: Gaussian σ, pad=ceil(3σ)
    Number of spots k ~ Uniform[0,maxk]
    
    This is definitely not realistic but good enough for testing
    TODO: add more realistic PSF shapes, maybe Airy disks?
    TODO: add correlated noise, background gradients
    """
    def __init__(self, length=60000, size=100, max_particles=4,
                 sigma=2.0, amp=100.0, bg=10.0, rng=None):
        self.len, self.size, self.maxk = length, size, max_particles
        self.sigma, self.amp, self.bg = sigma, amp, bg
        self.rng = rng or np.random.default_rng()
        # build Gaussian kernel (2D): exp(-(x^2+y^2)/(2σ^2))
        pad = int(math.ceil(3 * sigma))  # roughly covers 99.7% of the mass
        k = np.arange(-pad, pad+1)
        xx, yy = np.meshgrid(k, k)
        self.kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)
        self.kernel /= self.kernel.sum()  # normalize to unit mass
        self.pad = pad
        # print(f"kernel size: {self.kernel.shape}, sigma: {sigma}")  # debug
    def __len__(self): return self.len
    def _coords(self, k):
        # choose random centers avoiding edges - otherwise particles get cut off
        return [(self.rng.uniform(self.pad, self.size - self.pad),
                 self.rng.uniform(self.pad, self.size - self.pad))
                for _ in range(k)]
    def __getitem__(self, idx):
        k = self.rng.integers(0, self.maxk+1)  # how many particles
        img = np.full((self.size, self.size), self.bg, np.float32)  # background
        for cx, cy in self._coords(k):
            ix, iy = int(cx), int(cy)  # pixel coordinates
            y0, y1 = iy - self.pad, iy + self.pad + 1
            x0, x1 = ix - self.pad, ix + self.pad + 1
            # make sure we don't go out of bounds
            if 0 <= x0 < x1 <= self.size and 0 <= y0 < y1 <= self.size:
                img[y0:y1, x0:x1] += self.amp * self.kernel
        # add Poisson noise - this is the key for realism
        noisy = np.random.poisson(img).astype(np.float32)
        # normalize to [0,1] range for training stability
        maxv = noisy.max()
        if maxv > 0:
            noisy /= maxv
        # else: all zeros, just return as is
        return torch.from_numpy(noisy).unsqueeze(0), k

# mean/std estimation & transforms - probably could cache this but whatever
def dataset_mean_std(folder: Path, batches: int = 16, batch: int = 64) -> Tuple[float, float]:
    """rough estimate over a few batches - don't need to be super precise"""
    loader = DataLoader(ImageFolder(folder, transform=T.ToTensor()), batch_size=batch)
    n, mean, var = 0, 0.0, 0.0
    for i, (imgs, _) in enumerate(loader):
        if i >= batches: break  # limit compute, good enough approximation
        px = imgs.numel() / imgs.size(0)  # pixels per image
        mean += imgs.mean([0,2,3]) * px
        var  += imgs.var([0,2,3], False) * px
        n    += px
    mean = (mean / n)[0].item()  # extract scalar from tensor
    std  = torch.sqrt(var / n)[0].item()
    return mean, std


def make_transforms(mean: float, std: float, alpha: float):
    """data augmentation pipeline - more is usually better for small datasets"""
    train = [
        T.Grayscale(1),  # convert to grayscale if needed
        T.ToTensor(),
        T.Normalize((mean,), (std,)),  # normalize with dataset stats
        T.RandomHorizontalFlip(),  # flip left-right - particles don't have chirality
        T.RandomVerticalFlip(),    # and up-down, maybe unnecessary but cheap
        T.RandomRotation(180),     # rotate - again, particles are rotationally symmetric
        T.RandomAffine(0,
                       translate=(0.15,0.15),  # small translations
                       scale=(0.85,1.15)),     # slight zoom in/out
    ]
    if alpha > 0:
        try:
            from torchvision.transforms.v2 import MixUp
            train.append(MixUp(alpha=alpha))  # mixup for regularization - sometimes helps
        except ImportError:
            print('Warning: MixUp skipped (needs torchvision>=0.15)')
            # older torchvision doesn't have this, oh well
    train_tf = T.Compose(train)
    # validation transform: just normalize, no augmentation
    val_tf   = T.Compose([T.Grayscale(1), T.ToTensor(), T.Normalize((mean,), (std,))])
    return train_tf, val_tf

# run one epoch (train/eval) - this got messy but it works
def run_epoch(model, loader, criterion, opt=None, ordinal=False):
    """
    Run one epoch: if opt provided → train; else eval.
    Returns (loss, accuracy, per-class-acc)
    
    The ordinal flag is for CORAL models which need special handling
    """
    model.train(opt is not None)  # set mode based on whether we have optimizer
    corr = np.zeros(5, int)  # correct predictions per class
    tot  = np.zeros(5, int)  # total predictions per class
    loss_sum = 0.0
    for x, y in tqdm(loader, leave=False):  # leave=False so progress bars don't stack
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if ordinal:
            # CORAL needs special target encoding
            # convert class labels to cumulative binary targets
            levels = torch.zeros(y.size(0), logits.size(1), device=device)
            for i, lab in enumerate(y): levels[i, :lab] = 1  # ordinal levels
            loss = criterion(logits, levels)
        else:
            # standard cross-entropy
            loss = criterion(logits, y)
        if opt:
            # training step
            opt.zero_grad()
            loss.backward()
            opt.step()
        loss_sum += loss.item() * y.size(0)  # accumulate weighted loss
        # get predictions
        preds = coral_to_label(logits) if ordinal else logits.argmax(1)
        # accumulate per-class statistics
        for c in range(5):
            mask = (y == c)
            tot[c]  += mask.sum().item()
            corr[c] += ((preds == y) & mask).sum().item()
    acc = corr.sum() / tot.sum()  # overall accuracy
    # per-class accuracy, handle division by zero
    per = np.divide(corr, tot, out=np.zeros_like(corr, float), where=tot!=0)
    return loss_sum / len(loader.dataset), acc, per

# plot training histories - useful for comparing models
def plot_multi_history(histories: dict[str, dict[str, list[float]]]):
    """compare losses and accuracies across models - helps see which architecture works best"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # plot losses
    for tag, h in histories.items():
        ax1.plot(h['train_losses'], linestyle='-',  label=f'{tag} Train')
        ax1.plot(h['val_losses'],   linestyle='--', label=f'{tag} Val')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss Comparison')
    ax1.legend()
    # plot accuracies (convert to percentages)
    for tag, h in histories.items():
        ax2.plot([a*100 for a in h['train_accs']], linestyle='-',  label=f'{tag} Train')
        ax2.plot([a*100 for a in h['val_accs']],   linestyle='--', label=f'{tag} Val')
    ax2.set(xlabel='Epoch', ylabel='Accuracy (%)', title='Accuracy Comparison')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('training_history_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# training function - this is where the magic happens
def train_model(model, model_name, train_loader, val_loader, epochs, lr, gamma, ds_tag):
    """Train a single model and save with descriptive name
    
    TODO: add early stopping, learning rate scheduling improvements
    TODO: maybe add some regularization techniques
    """
    print(f"\nTraining {model_name}")
    print(f"   Parameters: {format_param_count(count_parameters(model))}")
    # could also print model size in MB: param_count * 4 / 1024**2
    
    # setup training components
    ordinal = isinstance(model, MobileNetCoral)  # check if using CORAL
    criterion = CoralLoss() if ordinal else FocalLoss(gamma)
    # tried different optimizers: SGD, Adam, AdamW - AdamW seems to work best
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Use get_last_lr() instead of verbose=True to avoid warning
    # step every epochs//3 seemed reasonable, could tune this
    # ensure step_size is at least 1 to avoid division by zero
    step_size = max(1, epochs//3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    model = model.to(device)
    best_val_acc = 0.0
    history = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}
    
    for epoch in range(epochs):
        # training phase
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, ordinal)
        
        # validation phase - no gradients needed
        with torch.no_grad():
            val_loss, val_acc, val_per_class = run_epoch(model, val_loader, criterion, None, ordinal)
        
        # update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]  # Use get_last_lr() instead of verbose
        
        # record history for plotting later
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        
        # print progress - convert to percentage for readability
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train: {train_loss:.4f}/{train_acc*100:.2f}% | "
              f"Val: {val_loss:.4f}/{val_acc*100:.2f}% | "
              f"LR: {current_lr:.2e}")
        
        # save best model with descriptive name - only save when validation improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"{model_name}-{ds_tag}-epoch{epoch+1}.pth"
            # save everything we might need later
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model_name': model_name,
                'param_count': count_parameters(model),
                # 'train_args': args,  # might want to add this later
            }, save_path)
            print(f"   Best model saved: {save_path} (acc: {best_val_acc*100:.2f}%)")
    
    print(f"Training completed for {model_name}. Best val acc: {best_val_acc*100:.2f}%")
    return history

# main script - the entry point
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train particle-count CNN (0–4)')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--data_root', help='folder w/ 0–4 subdirs')
    src.add_argument('--synthetic', type=int, help='generate N synthetic imgs')
    ap.add_argument('--train_dir'); ap.add_argument('--val_dir')  # for separate train/val dirs
    ap.add_argument('--arch', help='comma-separated: baseline,enhanced,mobilenet', default='mobilenet')
    ap.add_argument('--epochs', type=int, default=80)  # 80 usually enough, could go higher
    ap.add_argument('--batch',  type=int, default=64)  # tune based on GPU memory
    ap.add_argument('--lr',     type=float, default=1e-3)  # 1e-3 is a good starting point
    ap.add_argument('--gamma',  type=float, default=2.0)  # focal loss gamma
    ap.add_argument('--mixup',  type=float, default=0.0)  # mixup alpha, 0 = disabled
    ap.add_argument('--resume')  # not implemented yet but keeping for future
    ap.add_argument('--save_name', default='o3v2_best_model')  # legacy name
    args = ap.parse_args()

    # identify dataset tag for filenames - so we know what we trained on
    if args.data_root:
        ds_tag = Path(args.data_root).stem
    elif args.train_dir:
        ds_tag = Path(args.train_dir).stem
    elif args.val_dir:
        ds_tag = Path(args.val_dir).stem  # probably not the best choice but whatever
    else:
        ds_tag = 'synthetic'

    # load data - this part is a bit messy with all the different options
    if args.synthetic:
        # synthetic dataset - good for quick testing
        print(f"Using synthetic dataset with {args.synthetic} samples")
        full_dataset = SyntheticParticleDataset(length=args.synthetic)
        train_size = int(0.8 * len(full_dataset))  # 80/20 split
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2)
    else:
        # real dataset
        if args.data_root:
            # single directory with subdirs for each class
            data_path = Path(args.data_root)
            print(f"Loading data from {data_path}")
            # estimate dataset statistics - needed for normalization
            mean, std = dataset_mean_std(data_path)
            print(f"Dataset stats: mean={mean:.3f}, std={std:.3f}")
            train_tf, val_tf = make_transforms(mean, std, args.mixup)
            
            full_dataset = ImageFolder(data_path, transform=train_tf)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            # Apply different transforms to validation set - this is a bit hacky
            # TODO: cleaner way to do this without recreating the dataset
            val_dataset.dataset = ImageFolder(data_path, transform=val_tf)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2)
        else:
            # separate train/val directories - cleaner approach
            train_path = Path(args.train_dir)
            val_path = Path(args.val_dir)
            print(f"Using separate train/val dirs: {train_path}, {val_path}")
            # use training set for statistics
            mean, std = dataset_mean_std(train_path)
            train_tf, val_tf = make_transforms(mean, std, args.mixup)
            
            train_dataset = ImageFolder(train_path, transform=train_tf)
            val_dataset = ImageFolder(val_path, transform=val_tf)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2)

    # model architecture selection - can train multiple at once
    model_classes = {
        'baseline': ParticleCounterCNN,      # simple CNN
        'enhanced': EnhancedParticleCounterCNN,  # deeper CNN
        'mobilenet': MobileNetCoral          # pretrained + ordinal regression
    }
    
    architectures = [arch.strip() for arch in args.arch.split(',')]
    histories = {}  # store training curves for each model
    
    print(f"Dataset: {ds_tag}")
    print(f"Training architectures: {', '.join(architectures)}")
    
    # train each architecture - could parallelize this but usually memory-limited anyway
    for arch_name in architectures:
        if arch_name not in model_classes:
            print(f"Warning: Unknown architecture: {arch_name}, skipping...")
            continue
        
        # create model and generate name with parameter count
        model = model_classes[arch_name]()
        param_count = count_parameters(model)
        model_name = get_model_name_with_params(arch_name, param_count)
        # print(f"Model {model_name} has {param_count} parameters")  # debug
        
        # train model - this is where the actual work happens
        history = train_model(
            model, model_name, train_loader, val_loader, 
            args.epochs, args.lr, args.gamma, ds_tag
        )
        histories[model_name] = history
        
        # cleanup GPU memory - important when training multiple models
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # plot comparison if multiple models trained - helps choose the best one
    if len(histories) > 1:
        print("\nGenerating training history comparison...")
        plot_multi_history(histories)
    elif len(histories) == 1:
        print("Only one model trained, skipping comparison plot")
    
    print("\nAll training completed!")
    # could add some final statistics here: best model, final accuracies, etc.