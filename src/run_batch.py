from config import CONFIG
from train import main

for seed in (1, 2, 3, 4, 5):
    main({**CONFIG, "attn_after": None, "seed": seed})                          # CNN
    main({**CONFIG, "attn_after": 2, "seed": seed})                             # hybrid 2
    main({**CONFIG, "attn_after": 3, "seed": seed})                             # hybrid 3
    main({**CONFIG, "attn_after": 4, "seed": seed})                             # hybrid 4
    main({**CONFIG, "attn_after": 4, "attn_type": "conv", "seed": seed})        # control
    main({**CONFIG, "model": "vit", "seed": seed})                              # ViT