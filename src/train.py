import csv, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")                 # save plots to file, no window needed
import matplotlib.pyplot as plt

from config import CONFIG
from data import get_dataloaders
from model import build_model, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name):
    if name == "dml":                         # your AMD GPU
        import torch_directml
        return torch_directml.device()
    return torch.device(name)                 # "cpu" or "cuda"


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given, else just evaluates."""
    training = optimizer is not None
    model.train(training)                     # train(False) == eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):    # no gradients while evaluating
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                 # B x num_classes
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main(cfg):
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    if cfg["model"] == "vit":
        tag = "vit"
    else:
        kind = "hyb" if cfg["attn_type"] == "attention" else "ctrl"
        tag = "cnn" if cfg["attn_after"] is None else f"{kind}{cfg['attn_after']}"
    epochs_tag = "" if cfg["epochs"] == 30 else f"_e{cfg['epochs']}"
    run_name = f"{tag}_n{cfg['n_train_per_class']}_d{cfg['data_seed']}{epochs_tag}_s{cfg['seed']}"
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "final.pt").exists():     # finished run: don't redo it
        print(f"skipping {run_name}: already done")
        return
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    train_loader, val_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    print(f"run: {run_name} | device: {device} | parameters: {count_parameters(model):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, loss_fn, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device, loss_fn)
        scheduler.step()
        history.append([epoch, tr_loss, tr_acc, va_loss, va_acc])
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - t0:.1f}s")
        
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / "final.pt")   # needed for test evaluation later
    
    # save the log and the curves
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writerows(history)

    h = np.array(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h[:, 0], h[:, 1], label="train"); ax[0].plot(h[:, 0], h[:, 3], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h[:, 0], h[:, 2], label="train"); ax[1].plot(h[:, 0], h[:, 4], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)
    torch.save(model.state_dict(), out_dir / "final.pt") #for test evaluation
    print("saved to", out_dir)


if __name__ == "__main__":
    main(CONFIG)