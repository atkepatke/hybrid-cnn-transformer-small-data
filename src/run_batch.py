# src/run_batch.py
from config import CONFIG
from train import main
from evaluate import main as evaluate_all

# Part 1: 100-epoch supplementary check, ViT and CNN, size 500
for seed in (1, 2, 3):
    main({**CONFIG, "model": "vit", "epochs": 100, "seed": seed})
    main({**CONFIG, "attn_after": None, "epochs": 100, "seed": seed})

# Part 2: size 1000, all six models, 30 epochs
for seed in (1, 2, 3):
    for extra in (
        {"attn_after": None},                          # CNN
        {"attn_after": 2},                             # hybrid 2
        {"attn_after": 3},                             # hybrid 3
        {"attn_after": 4},                             # hybrid 4
        {"attn_after": 4, "attn_type": "conv"},        # control
        {"model": "vit"},                              # ViT
    ):
        main({**CONFIG, "n_train_per_class": 1000, "seed": seed, **extra})

# Test evaluation of the new runs (finished old runs are skipped automatically)
evaluate_all()