import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

groups = defaultdict(list)
print(f"{'run':<20}{'epochs':>7}{'train_acc':>11}{'val_acc':>9}{'val_loss':>10}")
for d in sorted(Path("results").iterdir()):
    f = d / "metrics.csv"
    if not f.exists():
        continue
    with open(f) as fh:
        last = list(csv.DictReader(fh))[-1]
    ep, tr, va, vl = int(last["epoch"]), float(last["train_acc"]), float(last["val_acc"]), float(last["val_loss"])
    print(f"{d.name:<20}{ep:>7}{tr:>11.3f}{va:>9.3f}{vl:>10.3f}")
    groups[d.name.rsplit("_s", 1)[0]].append((tr, va, vl))   # group = same model, all seeds

print("\nMean over seeds (± sample std)")
print(f"{'model':<18}{'runs':>5}{'val_acc':>16}{'val_loss':>10}{'gap':>8}")
for g, v in groups.items():
    tr, va, vl = zip(*v)
    sd = f"± {st.stdev(va):.3f}" if len(va) > 1 else "(1 run)"
    gap = st.mean(t - a for t, a in zip(tr, va))
    print(f"{g:<18}{len(v):>5}{st.mean(va):>9.3f} {sd:<7}{st.mean(vl):>10.3f}{gap:>8.3f}")