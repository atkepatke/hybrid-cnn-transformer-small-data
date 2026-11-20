# All experiment settings live here, so nothing is hardcoded in the model or data code.
CONFIG = {
    # data
    "dataset": "cifar10",
    "n_train_per_class": 500,   # 500 or 1000
    "data_seed": 1,             # 1, 2 or 3 (which saved split to use)
    "batch_size": 128,
    "num_classes": 10,
    "in_channels": 3,
    "image_size": 32,
    # model
    "widths": (32, 64, 128, 256),   # channels in stages 1-4
    "blocks_per_stage": 2,
    # hardware: "cpu", "dml" (your GPU), or "cuda" (Kaggle) - used later
    "device": "dml",
    # training
    "seed": 1,               # training seed (init + batch order)
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "attn_after": None,   # None = plain CNN; 2, 3 or 4 = attention at the end of that stage
    "attn_depth": 1,      # number of Transformer blocks
    "num_heads": 4,
    "attn_type": "attention",   # "attention" or "conv" (the control)
    "model": "cnn",        # "cnn" (covers cnn, hybrids, control) or "vit"
    "patch_size": 4,
    "vit_dim": 256,
    "vit_depth": 5,
}