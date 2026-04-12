# Phase 1 Setup & Run Guide

## What Each File Does

| File | Purpose | Run? |
|------|---------|------|
| `config_phase1.py` | All paths and hyperparameters | Edit only |
| `cache_embeddings.py` | Runs CLIP once, saves embeddings to disk | Run once |
| `adapter_model.py` | The lightweight model architecture | Import only |
| `dataset_phase1.py` | Loads cached embeddings during training | Import only |
| `train_phase1.py` | Main training script | Run to train |

---

## Step 0 — Install Dependencies

```bash
# Install PyTorch (CPU only, saves ~2GB vs GPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install CLIP (must use GitHub, not PyPI)
pip install git+https://github.com/openai/CLIP.git

# Install other dependencies
pip install Pillow tqdm numpy ftfy regex
```

---

## Step 1 — Edit config_phase1.py

Open `config_phase1.py` and set:

```python
DATA_ROOT    = "data"               # your dataset root
TASK_A_TRAIN = "data/task_a/train.json"
TASK_A_VAL   = "data/task_a/val.json"
IMAGES_DIR   = "data/images"        # folder containing all images
```

### Expected JSON Format

Your train.json / val.json should look like:

```json
[
    {
        "idiom":    "kick the bucket",
        "sentence": "He finally kicked the bucket after a long illness.",
        "images":   ["001_0.jpg", "001_1.jpg", "001_2.jpg", "001_3.jpg", "001_4.jpg"],
        "label":    2
    },
    ...
]
```

- `images`: list of image filenames, paths relative to `IMAGES_DIR`
- `label`: 0-based index of the correct image

If your format is different, edit the `load_dataset()` function in `cache_embeddings.py`.

---

## Step 2 — Cache Embeddings (Run Once)

```bash
python cache_embeddings.py
```

**What happens:**
- Downloads CLIP ViT-B/32 weights (~350 MB, first run only)
- Processes all images and text through CLIP
- Saves tensors to `cache/task_a_train.pt`, `cache/task_a_val.pt`
- Takes ~15-20 minutes (once only)
- After this, CLIP is never needed again during training

**Expected output:**
```
Loading CLIP ViT-B/32 on CPU...
✓ CLIP loaded. Parameters: 151,277,313
  All parameters frozen — CLIP will never be trained.

Processing Task A...
  Task A Train: 100%|████████| 60/60 [12:30<00:00]
  Task A Val:   100%|████████| 11/11 [02:15<00:00]

✓ Task A saved:
    cache/task_a_train.pt  (60 samples)
    cache/task_a_val.pt    (11 samples)

Caching complete!
Next step: python train_phase1.py --task a
```

---

## Step 3 — Train

```bash
# Task A only
python train_phase1.py --task a

# Task B only
python train_phase1.py --task b

# Both tasks
python train_phase1.py --task both
```

**What happens:**
- Loads cached embeddings (milliseconds)
- Trains only the adapter (~265K params)
- Each epoch: ~1-2 minutes on CPU
- Early stopping after 10 epochs without improvement
- Saves best model to `models/phase1/best_adapter_task_a.pt`

**Expected output:**
```
Training Phase 1 — Task A
===========================================================
  Samples:           60
  Candidates/sample: 5
  Augmentation:      ON (training mode)

  Label distribution:
    Label 0: 12 samples  ████████████
    Label 1: 13 samples  █████████████
    ...

  Model Parameter Count:
    Text Adapter:       132,737
    Image Adapter:      132,737
    Temperature:              1
    TOTAL:              265,475
    (We train 0.18% of CLIP's parameter count)

  Epoch | Train Loss | Train Acc |  Val Acc |    Best |       LR
  -----------------------------------------------------------------
       1 |     1.5832 |    22.5%  |   24.3%  |  24.3%  | 1.00e-03 ← BEST
       5 |     1.3241 |    31.2%  |   28.6%  |  28.6%  | 9.50e-04 ← BEST
      10 |     1.1053 |    38.4%  |   32.1%  |  32.1%  | 8.50e-04 ← BEST
      20 |     0.9821 |    45.2%  |   35.7%  |  35.7%  | 6.50e-04 ← BEST
      ...

  Best Validation Accuracy: 35.7%  (epoch 28)
  Baseline (IRFL CLIP):      ~22.0%
  Improvement:               +13.7 percentage points
```

---

## Troubleshooting

**"Cache file not found"**
→ Run `python cache_embeddings.py` first.

**"Could not load image: ..."**
→ Check that `IMAGES_DIR` is correct and image filenames in JSON match actual files.

**Accuracy stuck at ~20%**
→ Check label distribution — if all labels are 0, the dataset loading has a bug.
→ Run `python -c "from dataset_phase1 import *; d = CachedEmbeddingDataset('cache/task_a_train.pt'); d.get_class_distribution()"`

**Out of memory**
→ Reduce `BATCH_SIZE` to 4 in `config_phase1.py`

**Training too slow**
→ Make sure you ran `cache_embeddings.py` first. Without caching, each batch loads CLIP.

---

## After Phase 1

Once you see val accuracy at 30%+, move to Phase 2:
- Idiom-Aware Prompt Decomposition (IAPD)
- ConceptNet NumberBatch integration
- Back-translation augmentation

Target after Phase 2: **37–43%**
