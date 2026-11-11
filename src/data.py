import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from splits import load_split

# To support another dataset, add it here (and create its split files).
DATASETS = {"cifar10": CIFAR10, "cifar100": CIFAR100}


def _split_path(cfg):
    return f"splits/{cfg['dataset']}_n{cfg['n_train_per_class']}_d{cfg['data_seed']}.json"


def get_dataloaders(cfg):
    """Returns (train_loader, val_loader). Test data is NOT touched here."""
    DatasetClass = DATASETS[cfg["dataset"]]
    train_idx, val_idx, _ = load_split(_split_path(cfg))

    # 1) Mean/std from the training subset only (raw pixels, scaled to 0-1)
    raw = DatasetClass(root="data", train=True, download=True)
    pixels = raw.data[train_idx] / 255.0              # N x 32 x 32 x 3
    mean = pixels.mean(axis=(0, 1, 2)).tolist()
    std = pixels.std(axis=(0, 1, 2)).tolist()

    # 2) Two transforms: augmented for training, plain for evaluation
    train_tf = transforms.Compose([
        transforms.RandomCrop(cfg["image_size"], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 3) Same underlying images, different transforms, selected by saved indices
    train_set = Subset(DatasetClass(root="data", train=True, transform=train_tf), train_idx)
    val_set = Subset(DatasetClass(root="data", train=True, transform=eval_tf), val_idx)

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=0)
    return train_loader, val_loader


def get_test_loader(cfg):
    """Only call this for the final evaluation. Uses the same normalization recipe."""
    DatasetClass = DATASETS[cfg["dataset"]]
    train_idx, _, _ = load_split(_split_path(cfg))
    raw = DatasetClass(root="data", train=True, download=True)
    pixels = raw.data[train_idx] / 255.0
    mean = pixels.mean(axis=(0, 1, 2)).tolist()
    std = pixels.std(axis=(0, 1, 2)).tolist()
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_set = DatasetClass(root="data", train=False, transform=eval_tf)
    return DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)