# Complete Setup Guide — Idiom Capstone Project

## Prerequisites

- **Python 3.10 or 3.11** (3.12+ may have compatibility issues with some packages)
- **Git** (for installing CLIP from GitHub)
- **~2 GB disk space** (for dependencies + ConceptNet embeddings)

---

## Quick Start (Automated)

Run the provided batch script from inside the project folder:

```
cd C:\Users\Komal\Desktop\idiom-capstone
setup_all.bat
```

This will handle everything automatically. If you prefer to do it manually, follow the steps below.

---

## Manual Setup (Step by Step)

### Step 1 — Create a Virtual Environment

Open a terminal in `C:\Users\Komal\Desktop\idiom-capstone` and run:

```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### Step 2 — Install PyTorch (CPU)

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3 — Install OpenAI CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

### Step 4 — Install Remaining Dependencies

```bash
pip install Pillow>=9.0.0 tqdm>=4.65.0 numpy>=1.24.0 ftfy>=6.1.1 regex>=2023.0.0 flask
```

### Step 5 — Download ConceptNet NumberBatch

```bash
python setup_conceptnet.py
```

This downloads ~150 MB of English word embeddings and converts them to a fast-loading pickle file. Takes a few minutes depending on your internet speed.

### Step 6 — Run the Demo App

```bash
python app.py
```

Then open your browser to **http://127.0.0.1:5000**

---

## What the Demo Does

The app lets you:
1. Upload 5 images + a sentence containing an idiom
2. The system ranks images by figurative relevance using CLIP + ConceptNet + IAPD
3. Each image gets categorised as: Figurative / Literal / Partial Literal / Random / Distractor
4. A full 7-section visual commonsense reasoning chain is generated

---

## Optional: Train the Models (For Full Accuracy)

The demo works in zero-shot mode without trained models, but for best results:

### Train Phase 4 Models

This requires the AdMIRe / SemEval-2025 dataset (not included in the repo). If you have the dataset:

```bash
python cache_phase3.py
python train_phase4.py --task a
python train_phase4.py --task b
```

Checkpoints are saved to `models/phase4_task_a_mlp.pt` and `models/phase4_task_b_mlp.pt`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install CLIP` fails | Make sure you use `pip install git+https://github.com/openai/CLIP.git` (not PyPI) |
| `ModuleNotFoundError: clip` | Activate the venv first: `venv\Scripts\activate` |
| ConceptNet download fails | Manually download from the URL printed in the error and place in `conceptnet/` |
| `flask` not found | Run `pip install flask` |
| Port 5000 in use | Kill other processes on port 5000, or edit `app.py` last line to change port |
