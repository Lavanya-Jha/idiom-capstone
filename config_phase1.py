"""
config_phase1.py
================
All paths and hyperparameters — pre-configured for YOUR Capstone dataset.

Your dataset structure (auto-detected):
    Capstone/
    ├── Subtask A/EN/
    │   ├── Train/train/
    │   │   ├── subtask_a_train.tsv     ← 70 training idioms
    │   │   └── {idiom}/               ← images per idiom
    │   └── Dev/
    │       ├── subtask_a_dev.tsv       ← 15 validation idioms
    │       └── {idiom}/               ← images per idiom
    └── Subtask B/
        ├── Train/
        │   ├── subtask_b_train.tsv     ← 22 training idioms
        │   └── {idiom}/
        └── Dev/
            ├── subtask_b_dev.tsv       ← 5 validation idioms
            └── {idiom}/

HOW TO RUN (no editing needed — paths are already correct):
    Step 1: python cache_embeddings.py
    Step 2: python train_phase1.py --task a
"""

import os

# ==============================================================================
# ROOT — set to the parent folder containing "Subtask A" and "Subtask B"
# Using ".." because this file lives inside Capstone/phase1/
# and the data lives in Capstone/
# ==============================================================================

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..")

# ==============================================================================
# TASK A PATHS
# ==============================================================================

TASK_A_TRAIN     = os.path.join(DATA_ROOT, "Subtask A", "EN", "Train", "train", "subtask_a_train.tsv")
TASK_A_VAL       = os.path.join(DATA_ROOT, "Subtask A", "EN", "Dev",   "subtask_a_dev.tsv")

# Image folders — images are stored in subfolders named after each idiom
TASK_A_TRAIN_IMG = os.path.join(DATA_ROOT, "Subtask A", "EN", "Train", "train")
TASK_A_VAL_IMG   = os.path.join(DATA_ROOT, "Subtask A", "EN", "Dev")

# ==============================================================================
# TASK B PATHS
# ==============================================================================

TASK_B_TRAIN     = os.path.join(DATA_ROOT, "Subtask B", "Train", "subtask_b_train.tsv")
TASK_B_VAL       = os.path.join(DATA_ROOT, "Subtask B", "Dev",   "subtask_b_dev.tsv")

TASK_B_TRAIN_IMG = os.path.join(DATA_ROOT, "Subtask B", "Train")
TASK_B_VAL_IMG   = os.path.join(DATA_ROOT, "Subtask B", "Dev")

# ==============================================================================
# OUTPUT FOLDERS (auto-created when you run the scripts)
# ==============================================================================

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "cache")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models", "phase1")

# ==============================================================================
# MODEL SETTINGS — Do NOT change
# ==============================================================================

CLIP_MODEL  = "ViT-B/32"   # Best CLIP variant for CPU — 600 MB RAM usage
EMBED_DIM   = 512           # Fixed output dim for ViT-B/32
ADAPTER_DIM = 64            # Middle ground: large enough to learn, small enough not to overfit
                            # ~66K params total

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

BATCH_SIZE          = 4     # Even smaller batch — more gradient updates per epoch on tiny data
EPOCHS              = 150   # More epochs with smaller, careful steps
LR                  = 3e-4  # Middle ground: not too fast (overfit) not too slow (no learning)
WEIGHT_DECAY        = 1e-3  # Mild regularisation — was 1e-2 which was too strong
TEMPERATURE         = 0.07
EARLY_STOP_PATIENCE = 30    # More patience — val set is only 15 samples so it fluctuates

# ==============================================================================
# TASK SETTINGS
# ==============================================================================

TASK_A_NUM_CANDIDATES = 5   # 5 candidate images per idiom in Task A
TASK_B_NUM_CANDIDATES = 4   # 4 candidate images per idiom in Task B

# ==============================================================================
# MISC
# ==============================================================================

SEED        = 42
NUM_WORKERS = 0             # Keep at 0 for CPU
LOG_EVERY   = 5
