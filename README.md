# Hybrid CNN–Transformer Classifier for Small Datasets

University Deep Learning course project. Investigates where self-attention should be inserted into a small CNN when training data is limited, compared with a plain CNN, a parameter-matched convolutional control, and a small Vision Transformer (ViT).

**Status: work in progress.** Final results will be added after the full experiment batch.

## Setup
- Python 3.10, PyTorch via `torch-directml` (AMD GPU on Windows). The code also runs on `cpu` or `cuda` (set `"device"` in `src/config.py`).
- `pip install -r requirements.txt`

## Layout
- `src/splits.py` – creates stratified, disjoint train/validation index files (`splits/`)
- `src/config.py` – all experiment settings
- `src/data.py` – data loading and augmentation (training data only)
- `src/model.py`, `src/attention.py`, `src/vit.py` – CNN, attention blocks, control, ViT
- `src/train.py` – training loop; writes config, metrics and curves to `results/<run>/`
- `src/run_batch.py`, `src/summarize.py` – run several experiments; aggregate over seeds
- `src/check_data.py`, `src/check_shapes.py`, `src/check_vit.py` – sanity checks

## Experimental Design
- Dataset: CIFAR-10, subsampled to 500 (and 1000) training images per class; 100 per class for validation; official test set reserved for final evaluation.
- Models: CNN baseline, CNN + attention after stage 2/3/4, parameter-matched conv control, small ViT.
- All models trained from scratch with the same recipe (AdamW, cosine schedule, 30 epochs) and the same split files; several training seeds per model.
- Last-epoch metrics are reported (no best-epoch selection).

## Run
```bash
python src/splits.py        # once, creates splits/
python src/train.py         # one run, per src/config.py
python src/run_batch.py     # several runs
python src/summarize.py     # table over seeds