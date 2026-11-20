# src/evaluate.py  -- run ONCE on the final models; test results must not guide decisions
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, f1_score

from config import CONFIG
from data import get_test_loader
from model import build_model
from train import get_device


def evaluate_run(run_dir, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x, y in get_test_loader(cfg):
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()

    cm = confusion_matrix(labels, preds, labels=list(range(cfg["num_classes"])))
    return {
        "test_acc": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "per_class_acc": (cm.diagonal() / cm.sum(axis=1)).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def main():
    device = get_device(CONFIG["device"])
    groups = defaultdict(list)
    for run_dir in sorted(Path("results").iterdir()):
        if not (run_dir / "final.pt").exists():
            continue
        out_file = run_dir / "test_metrics.json"
        if out_file.exists():                              # evaluate each model only once
            res = json.loads(out_file.read_text())
        else:
            res = evaluate_run(run_dir, device)
            out_file.write_text(json.dumps(res))
            print("evaluated", run_dir.name)
        groups[run_dir.name.rsplit("_s", 1)[0]].append(res)

    print(f"\n{'model':<18}{'runs':>5}{'test_acc':>18}{'macro_f1':>18}")
    for g, rs in groups.items():
        acc = [r["test_acc"] for r in rs]
        f1 = [r["macro_f1"] for r in rs]
        sd = lambda v: st.stdev(v) if len(v) > 1 else float("nan")
        print(f"{g:<18}{len(rs):>5}{st.mean(acc):>11.3f} ± {sd(acc):.3f}{st.mean(f1):>11.3f} ± {sd(f1):.3f}")


if __name__ == "__main__":
    main()