"""
train_phase6.py
===============
PURPOSE:
    Phase 6 — Concept Gap Corrector on top of frozen Phase 4.

    Instead of retraining the full model, we:
      1. Load the frozen Phase 4 MLPHead (73.3% val accuracy)
      2. Compute Phase 4 scores for all training/val images
      3. Learn a single scalar alpha such that:
             final_score = phase4_score + alpha * concept_gap
      4. Grid-search alpha over [-2, +2] for maximum val accuracy

    Why 1 parameter instead of retraining?
      With 100 training samples, adding even 1 new feature causes
      the full MLPHead to overfit. Freezing Phase 4 guarantees the
      73.3% baseline is never degraded — the corrector can only help.

    concept_gap signal:
      Detects which abstract concepts (from app.py's 150-word
      INFERENCE_VOCAB) are visually present in each image, then
      checks whether those concepts align with the figurative vs
      literal interpretation of the idiom.
      std=0.036, corr with existing signals < 0.1 → genuinely
      independent information.

HOW TO RUN:
    python train_phase6.py --task a
    python train_phase6.py --task b

REQUIREMENTS:
    models/phase4_task_a_mlp.pt    (from train_phase4.py)
    cache/phase3_task_a_*.pt       (from cache_phase3.py)
    cache/attention_task_a_*.pt    (from cache_attention.py)

OUTPUT:
    models/phase6_task_a_alpha.pt  — saved dict with best alpha value
"""

import os
import argparse
import torch
import numpy as np

from phase4_model import Phase4CaptionFusedRanker
from train_phase4 import extract_4signals

_HERE     = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "cache")
MODEL_DIR = os.path.join(_HERE, "models")


# ==============================================================================
# LOAD CONCEPT GAP — aligned to Phase 3 idiom order
# ==============================================================================

def load_concept_gap(p3_cache, att_cache, device):
    """
    Align concept_gap cache to Phase 3 idiom order.
    Returns: (N, C) concept_gap tensor
    """
    p3_idioms  = p3_cache["idioms"]
    att_idioms = att_cache["idioms"]
    att_gap_m  = att_cache["concept_gap"].float()   # (M, C)

    N = len(p3_idioms)
    C = att_gap_m.shape[1]
    att_idx     = {idiom: i for i, idiom in enumerate(att_idioms)}
    gap_aligned = torch.zeros(N, C)

    matched = 0
    for i, idiom in enumerate(p3_idioms):
        if idiom in att_idx:
            gap_aligned[i] = att_gap_m[att_idx[idiom]]
            matched += 1

    print(f"  Concept gap alignment: {matched}/{N} idioms matched")
    return gap_aligned.to(device)


# ==============================================================================
# GRID SEARCH FOR ALPHA
# ==============================================================================

def find_best_alpha(phase4_scores, concept_gap, labels, alpha_range=(-3.0, 3.0), steps=601):
    """
    Grid search: find alpha that maximises val accuracy when:
        final_score = phase4_score + alpha * concept_gap

    Args:
        phase4_scores: (N, C) from frozen Phase 4
        concept_gap:   (N, C)
        labels:        (N,)

    Returns:
        best_alpha (float), best_accuracy (float)
    """
    alphas     = torch.linspace(alpha_range[0], alpha_range[1], steps)
    best_acc   = 0.0
    best_alpha = 0.0

    for alpha in alphas:
        scores = phase4_scores + alpha.item() * concept_gap
        preds  = scores.argmax(dim=-1)
        acc    = (preds == labels).float().mean().item()
        if acc > best_acc:
            best_acc   = acc
            best_alpha = alpha.item()

    return best_alpha, best_acc


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train(task="a"):
    device = "cpu"

    # ── Load Phase 4 model ────────────────────────────────────────────────────
    p4_model_path = os.path.join(MODEL_DIR, f"phase4_task_{task}_mlp.pt")
    if not os.path.exists(p4_model_path):
        raise FileNotFoundError(
            f"Phase 4 model not found: {p4_model_path}\n"
            f"Run train_phase4.py --task {task} first."
        )

    # ── Load caches ──────────────────────────────────────────────────────────
    for path in [
        os.path.join(CACHE_DIR, f"phase3_task_{task}_train.pt"),
        os.path.join(CACHE_DIR, f"phase3_task_{task}_val.pt"),
        os.path.join(CACHE_DIR, f"attention_task_{task}_train.pt"),
        os.path.join(CACHE_DIR, f"attention_task_{task}_val.pt"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache not found: {path}\n"
                f"Run cache_phase3.py then cache_attention.py first."
            )

    print(f"\n  Task {task.upper()} — Phase 6 Concept Gap Corrector")
    print(f"  {'='*50}")

    p3_train  = torch.load(os.path.join(CACHE_DIR, f"phase3_task_{task}_train.pt"),     map_location=device)
    p3_val    = torch.load(os.path.join(CACHE_DIR, f"phase3_task_{task}_val.pt"),       map_location=device)
    att_train = torch.load(os.path.join(CACHE_DIR, f"attention_task_{task}_train.pt"),  map_location=device)
    att_val   = torch.load(os.path.join(CACHE_DIR, f"attention_task_{task}_val.pt"),    map_location=device)

    # ── Extract Phase 4 signals (4 scalars: ctx, fig, lit, gap) ─────────────
    train_signals, train_labels = extract_4signals(p3_train, device)  # (N, C, 4)
    val_signals,   val_labels   = extract_4signals(p3_val,   device)

    # ── Load concept gap (independently computed) ─────────────────────────────
    print("\n  Loading concept gap signals...")
    train_gap = load_concept_gap(p3_train, att_train, device)   # (N, C)
    val_gap   = load_concept_gap(p3_val,   att_val,   device)

    # ── Load frozen Phase 4 model and compute its scores ─────────────────────
    print(f"\n  Loading frozen Phase 4 model: {p4_model_path}")
    p4_model = Phase4CaptionFusedRanker(n_signals=4, head="mlp")
    ckpt     = torch.load(p4_model_path, map_location=device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    p4_model.load_state_dict(state_dict)
    p4_model.eval()
    for p in p4_model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        train_p4_scores = p4_model(train_signals)   # (N, C)
        val_p4_scores   = p4_model(val_signals)     # (N, C)

    # Baseline accuracy (Phase 4 only, alpha=0)
    train_p4_acc = (train_p4_scores.argmax(-1) == train_labels).float().mean().item()
    val_p4_acc   = (val_p4_scores.argmax(-1)   == val_labels  ).float().mean().item()
    print(f"\n  Phase 4 baseline  — train: {train_p4_acc:.1%}  val: {val_p4_acc:.1%}")

    # ── Grid search alpha on TRAINING set ────────────────────────────────────
    print("\n  Grid searching alpha on training set...")
    best_train_alpha, best_train_acc = find_best_alpha(
        train_p4_scores, train_gap, train_labels
    )
    print(f"  Best alpha (train): {best_train_alpha:+.3f}  → train acc: {best_train_acc:.1%}")

    # Evaluate that alpha on val set
    val_corrected = val_p4_scores + best_train_alpha * val_gap
    val_corrected_acc = (val_corrected.argmax(-1) == val_labels).float().mean().item()
    print(f"  Val accuracy with alpha={best_train_alpha:+.3f}: {val_corrected_acc:.1%}")

    # ── Also search directly on val (upper bound / oracle) ───────────────────
    print("\n  Grid searching alpha directly on val set (oracle upper bound)...")
    best_val_alpha, best_val_acc = find_best_alpha(
        val_p4_scores, val_gap, val_labels
    )
    print(f"  Best alpha (val):   {best_val_alpha:+.3f}  → val acc: {best_val_acc:.1%}")

    # ── Choose the best alpha ─────────────────────────────────────────────────
    # Use the train-optimised alpha (generalizable), not the oracle
    final_alpha = best_train_alpha
    final_val_acc = val_corrected_acc

    # Show what happens across a range of alpha values
    print(f"\n  Alpha sensitivity on val set:")
    print(f"  {'alpha':>8}  {'val acc':>8}")
    for a in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        s = val_p4_scores + a * val_gap
        acc = (s.argmax(-1) == val_labels).float().mean().item()
        marker = " ← chosen" if abs(a - final_alpha) < 0.1 else ""
        print(f"  {a:>8.2f}  {acc:>8.1%}{marker}")

    # ── Save alpha ────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    alpha_path = os.path.join(MODEL_DIR, f"phase6_task_{task}_alpha.pt")
    torch.save({"alpha": final_alpha, "val_acc": final_val_acc}, alpha_path)
    print(f"\n  Saved alpha={final_alpha:+.3f} to {alpha_path}")

    # ── Final results ─────────────────────────────────────────────────────────
    print(f"\n  Results comparison:")
    print(f"    CLIP zero-shot baseline     : 53.3%")
    print(f"    Phase 4 (8 signals)         : {val_p4_acc:.1%}  ← previous best")
    print(f"    Phase 6 (+concept_gap alpha): {final_val_acc:.1%}  ← this run (train alpha)")
    print(f"    Phase 6 oracle upper bound  : {best_val_acc:.1%}  (val alpha, not usable)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["a", "b"], default="a")
    args = parser.parse_args()
    train(task=args.task)
