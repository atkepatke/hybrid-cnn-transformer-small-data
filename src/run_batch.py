from config import CONFIG
from train import main

for seed in (1, 2, 3):
    main({**CONFIG, "model": "vit", "seed": seed})