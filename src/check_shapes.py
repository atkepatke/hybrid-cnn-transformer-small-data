# src/check_shapes.py
import torch
from config import CONFIG
from model import build_model, count_parameters

cases = [(None, "attention"), (2, "attention"), (3, "attention"),
         (4, "attention"), (4, "conv")]

for attn_after, attn_type in cases:
    cfg = {**CONFIG, "attn_after": attn_after, "attn_type": attn_type}
    model = build_model(cfg).eval()
    x = torch.randn(2, cfg["in_channels"], cfg["image_size"], cfg["image_size"])
    shapes = []
    h = model.stem(x)
    for stage in model.stages:
        h = stage(h)
        shapes.append(tuple(h.shape))
    logits = model(x)
    print(f"attn_after={attn_after} ({attn_type}): params={count_parameters(model):,} logits={tuple(logits.shape)}")
    print("   stage outputs:", shapes)