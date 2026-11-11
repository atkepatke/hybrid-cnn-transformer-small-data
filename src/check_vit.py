import torch
from config import CONFIG
from model import build_model, count_parameters

cfg = {**CONFIG, "model": "vit"}
model = build_model(cfg).eval()
x = torch.randn(2, 3, cfg["image_size"], cfg["image_size"])

p = model.patch_embed(x)
print("patch embedding :", tuple(p.shape))
t = p.flatten(2).transpose(1, 2)
print("tokens          :", tuple(t.shape))
print("after blocks    :", tuple(model.blocks(t + model.pos_embed).shape))
print("logits          :", tuple(model(x).shape))
print("parameters      :", f"{count_parameters(model):,}")