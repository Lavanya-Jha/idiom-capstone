"""
train_phase4.py
===============
PURPOSE:
    Train the Phase 4 CaptionFusedRanker on the Phase 3 cache.

    Adaptations from the Subtask B training pipeline (idiom_classifier/train.py):
      ✓ AdamW optimizer
      ✓ ReduceLROnPlateau scheduler (from Subtask B)
      ✓ Early stopping with patience
      ✓ Pairwise MarginRankingLoss (more suited to ranking than CrossEntropy)

    Key differences from Phase 2:
      • Uses Phase 3 cache (includes caption embeddings)
      • Trains the MLPHead / LinearScorer on 8 scalar signals
      • Uses pairwise ranking loss instead of CrossEntropy
      • Saves best model checkpoint to models/

HOW TO RUN:
    python train_phase4.py --task a
    python train_phase4.py --task a --head linear   # ultra-minimal (10 params)
    python train_phase4.py --task both
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from phase4_model import (
    Phase4CaptionFusedRanker,
    extract_signals,
    pairwise_ranking_loss,
)
import config_phase1 as cfg

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "cache")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# HELPERS
# ==============================================================================

def load_cache(task, split):
    path = os.path.join(CACHE_DIR, f"phase3_task_{task}_{split}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def accuracy_from_signals(model, signals, labels):
    model.eval()
    with torch.no_grad():
        preds = model.predict(signals)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


# ==============================================================================
# TRAINING LOOP  (adapted from Subtask B train.py)
# ==============================================================================

def train_task(task, head='mlp', epochs=500, lr=1e-2, patience=60,
               margin=0.05, weight_decay=0.0):
    print(f"\n{'='*60}")
    print(f"  Training Phase 4  |  Task {task.upper()}  |  head={head}")
    print(f"{'='*60}")

    train_cache = load_cache(task, "train")
    val_cache   = load_cache(task, "val")

    if train_cache is None:
        print(f"  [SKIP] No Phase 3 cache found for Task {task.upper()} train.")
        return None

    train_signals, train_labels = extract_signals(train_cache, DEVICE)
    print(f"  Train: {train_signals.shape[0]} samples, {train_signals.shape[1]} images each")

    val_signals, val_labels = None, None
    if val_cache is not None:
        val_signals, val_labels = extract_signals(val_cache, DEVICE)
        print(f"  Val:   {val_signals.shape[0]} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Phase4CaptionFusedRanker(n_signals=8, head=head).to(DEVICE)
    model.count_parameters()

    # ── Optimizer / Scheduler (adapted from Subtask B) ─────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=20, min_lr=1e-5)

    best_val_acc = 0.0
    best_epoch   = 0
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        # ── Training step ─────────────────────────────────────────────────
        model.train()
        scores = model(train_signals)                           # (N, C)
        loss   = pairwise_ranking_loss(scores, train_labels, margin)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (stability, from Subtask B)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Evaluation ────────────────────────────────────────────────────
        train_acc = accuracy_from_signals(model, train_signals, train_labels)
        val_acc   = accuracy_from_signals(model, val_signals, val_labels) \
                    if val_signals is not None else 0.0

        scheduler.step(val_acc)

        # ── Logging ───────────────────────────────────────────────────────
        if epoch % 20 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Ep {epoch:4d} | loss={loss.item():.4f} | "
                  f"train={train_acc*100:.1f}% | val={val_acc*100:.1f}% | lr={lr_now:.2e}")

        # ── Early stopping + checkpoint ───────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            ckpt_path = os.path.join(MODELS_DIR, f"phase4_task_{task}_{head}.pt")
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_acc":      val_acc,
                "head":         head,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\n  ✓ Best val accuracy: {best_val_acc*100:.1f}%  (epoch {best_epoch})")
    print(f"  ✓ Saved: models/phase4_task_{task}_{head}.pt")
    return best_val_acc


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default="a", choices=["a", "b", "both"])
    parser.add_argument("--head",   default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--epochs", type=int,   default=500)
    parser.add_argument("--lr",     type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--margin",   type=float, default=0.05)
    args = parser.parse_args()

    tasks = ["a", "b"] if args.task == "both" else [args.task]

    results = {}
    for t in tasks:
        acc = train_task(
            task=t,
            head=args.head,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            margin=args.margin,
        )
        if acc is not None:
            results[t] = acc

    print(f"\n{'='*60}")
    print("  PHASE 4 TRAINING SUMMARY")
    print(f"{'='*60}")
    for t, acc in results.items():
        print(f"  Task {t.upper()}: {acc*100:.1f}% val accuracy")

    print("""
  Baseline comparison:
    Phase 1 (CLIP zero-shot):  53.3%
    Phase 2 (IAPD + CN):       66.7%  ← previous best
    Phase 4 target:            > 66.7% (caption-fused)
""")
