from config import CONFIG
from train import main
from evaluate import main as evaluate_all

# Priority 1: attention vs control when properly trained (500/class, 100 epochs)
for seed in (1, 2, 3):
    main({**CONFIG, "attn_after": 4, "epochs": 100, "seed": seed})                          # hybrid 4
    main({**CONFIG, "attn_after": 4, "attn_type": "conv", "epochs": 100, "seed": seed})     # control
evaluate_all()      # test results for Part 1 are saved even if Part 2 is interrupted

# Priority 2: best-case data and training budget (1000/class, 100 epochs)
for seed in (1, 2, 3):
    main({**CONFIG, "model": "vit", "n_train_per_class": 1000, "epochs": 100, "seed": seed})
    main({**CONFIG, "attn_after": None, "n_train_per_class": 1000, "epochs": 100, "seed": seed})
evaluate_all()