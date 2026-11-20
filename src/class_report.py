# src/class_report.py -- per-class precision/recall/F1 from the saved TEST confusion matrices
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
GROUPS = ["cnn_n500_d1", "hyb2_n500_d1", "hyb3_n500_d1",
          "hyb4_n500_d1", "ctrl4_n500_d1", "vit_n500_d1"]


def per_class_metrics(cm):
    """cm[i][j] = number of images of true class i predicted as class j."""
    cm = np.array(cm, dtype=float)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp            # class c images predicted as something else
    fp = cm.sum(axis=0) - tp            # other images predicted as class c
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)             # = per-class accuracy
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


runs = defaultdict(list)
for d in sorted(Path("results").iterdir()):
    f = d / "test_metrics.json"
    if f.exists():
        runs[d.name.rsplit("_s", 1)[0]].append(json.loads(f.read_text()))

Path("figures").mkdir(exist_ok=True)
stats = {}                              # group -> (precision, recall, f1), each averaged over seeds
for g in GROUPS:
    if g not in runs:
        print("no results for", g)
        continue
    prf = [per_class_metrics(r["confusion_matrix"]) for r in runs[g]]
    stats[g] = tuple(np.mean([x[k] for x in prf], axis=0) for k in range(3))

    cm = np.mean([np.array(r["confusion_matrix"]) for r in runs[g]], axis=0)
    cm = cm / cm.sum(axis=1, keepdims=True)              # each row sums to 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=90)
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(g)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(f"figures/confusion_{g}.png", dpi=150); plt.close(fig)


def print_table(title, k):
    print(f"\n{title} (mean over seeds)")
    print(f"{'class':<12}" + "".join(f"{g.split('_')[0]:>8}" for g in stats))
    for i, c in enumerate(CLASSES):
        print(f"{c:<12}" + "".join(f"{stats[g][k][i]:>8.3f}" for g in stats))
    print(f"{'macro avg':<12}" + "".join(f"{stats[g][k].mean():>8.3f}" for g in stats))


print_table("Per-class recall = per-class accuracy", 1)
print_table("Per-class precision", 0)
print_table("Per-class F1", 2)

# One fully worked example: CNN, first seed, class "cat"
g = GROUPS[0]
cm = np.array(runs[g][0]["confusion_matrix"], dtype=float)
c = CLASSES.index("cat")
tp, fn, fp = cm[c, c], cm[c].sum() - cm[c, c], cm[:, c].sum() - cm[c, c]
p, r = tp / (tp + fp), tp / (tp + fn)
print(f"\nWorked example: {g}, first seed, class 'cat'")
print(f"TP={tp:.0f}  FN={fn:.0f}  FP={fp:.0f}")
print(f"precision = TP/(TP+FP) = {tp:.0f}/{tp + fp:.0f} = {p:.4f}")
print(f"recall    = TP/(TP+FN) = {tp:.0f}/{tp + fn:.0f} = {r:.4f}")
print(f"F1 = 2*P*R/(P+R) = {2 * p * r / (p + r):.4f}   (also 2TP/(2TP+FP+FN) = {2 * tp / (2 * tp + fp + fn):.4f})")

# Self-check against the macro-F1 that evaluate.py stored (sklearn)
mine = np.mean([per_class_metrics(r["confusion_matrix"])[2].mean() for r in runs[g]])
stored = np.mean([r["macro_f1"] for r in runs[g]])
print(f"\nCheck, macro-F1 for {g}: from confusion matrices = {mine:.4f}, stored = {stored:.4f}")