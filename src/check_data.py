from config import CONFIG
from data import get_dataloaders
import torch

train_loader, val_loader = get_dataloaders(CONFIG)

images, labels = next(iter(train_loader))
print("batch images:", tuple(images.shape))
print("batch labels:", tuple(labels.shape))
print("normalized mean ~", round(images.mean().item(), 3),
      " std ~", round(images.std().item(), 3))

for name, loader in [("train", train_loader), ("val", val_loader)]:
    counts = torch.zeros(CONFIG["num_classes"], dtype=torch.long)
    for _, y in loader:
        counts += torch.bincount(y, minlength=CONFIG["num_classes"])
    print(name, "class counts:", counts.tolist())