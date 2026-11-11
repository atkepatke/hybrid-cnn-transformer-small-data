# src/splits.py
import json
from pathlib import Path
import numpy as np


def make_split(labels, n_train_per_class, n_val_per_class, data_seed):
    """Stratified, disjoint train/val index lists from a labelled pool."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(data_seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)                       # same seed -> same order
        val_idx.extend(idx[:n_val_per_class])  # validation slice first
        train_idx.extend(idx[n_val_per_class:n_val_per_class + n_train_per_class])
    return sorted(int(i) for i in train_idx), sorted(int(i) for i in val_idx)


def check_split(labels, train_idx, val_idx, num_classes):
    """Fail loudly if the split breaks our rules."""
    labels = np.asarray(labels)
    assert not set(train_idx) & set(val_idx), "train/val overlap!"
    for name, idx in [("train", train_idx), ("val", val_idx)]:
        counts = np.bincount(labels[idx], minlength=num_classes)
        assert counts.min() == counts.max(), f"{name} not balanced: {counts}"


def save_split(path, train_idx, val_idx, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta, "train": train_idx, "val": val_idx}, f)


def load_split(path):
    with open(path) as f:
        d = json.load(f)
    return d["train"], d["val"], d["meta"]


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10

    pool = CIFAR10(root="data", train=True, download=True)  # official 50k pool
    labels = pool.targets

    for data_seed in (1, 2, 3):
        for n_train in (500, 1000):
            tr, va = make_split(labels, n_train, 100, data_seed)
            check_split(labels, tr, va, num_classes=10)
            save_split(f"splits/cifar10_n{n_train}_d{data_seed}.json", tr, va,
                       {"n_train_per_class": n_train, "n_val_per_class": 100,
                        "data_seed": data_seed})
            print(f"d{data_seed} n{n_train}: train={len(tr)} val={len(va)}")

    # nesting check: the 500 set should sit inside the 1000 set
    t500, _, _ = load_split("splits/cifar10_n500_d1.json")
    t1000, _, _ = load_split("splits/cifar10_n1000_d1.json")
    print("500 subset of 1000:", set(t500) <= set(t1000))