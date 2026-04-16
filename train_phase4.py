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
import itertools
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

def extract_4signals(cache, device="cpu"):
    """
    Extract only the 4 discriminative signals from Phase 3 cache.
    Drops the 4 caption-based signals which are identical per idiom
    when synthetic captions are used (adding noise, not signal).

    Signals:
        [0] sim(ctx,  image)  — contextual text vs image
        [1] sim(fig,  image)  — figurative text vs image  (now uses IDIOM_DB!)
        [2] sim(lit,  image)  — literal text vs image     (now uses IDIOM_DB!)
        [3] fig_lit_gap       — fig_sim - lit_sim         (most discriminative)
    """
    iapd = cache["iapd_embeddings"].float().to(device)   # (N, 3, 512)
    imgs = cache["image_embeddings"].float().to(device)  # (N, C, 512)
    gap  = cache["fig_lit_gap"].float().to(device)       # (N, C)
    lbls = cache["labels"].long().to(device)

    ctx = iapd[:, 2, :].unsqueeze(1)
    fig = iapd[:, 1, :].unsqueeze(1)
    lit = iapd[:, 0, :].unsqueeze(1)

    s_ctx = (ctx * imgs).sum(dim=-1)   # [0]
    s_fig = (fig * imgs).sum(dim=-1)   # [1]
    s_lit = (lit * imgs).sum(dim=-1)   # [2]
    # gap                               # [3]

    signals = torch.stack([s_ctx, s_fig, s_lit, gap], dim=-1)  # (N, C, 4)
    return signals, lbls


def grid_search_weights(signals, labels, steps=21):
    """
    Grid search over (w_ctx, w_fig, w_lit, w_gap) to find the best linear
    combination score = w0*ctx + w1*fig + w2*lit + w3*gap.

    Weights are non-negative and normalised to sum to 1.
    steps=21 → 0.0, 0.05, 0.10, ... 1.0 per weight (exhaustive but fast).
    Returns best_weights (4,), best_train_acc.
    """
    vals = [i / (steps - 1) for i in range(steps)]  # [0.0, 0.05, ..., 1.0]
    best_acc = -1.0
    best_w   = (0.25, 0.25, 0.25, 0.25)

    # Only enumerate combinations where weights sum to ~1.0
    # Use 3-way split (w0, w1, w3) and derive w2 = 1 - w0 - w1 - w3
    for w0, w1, w3 in itertools.product(vals, vals, vals):
        w2 = round(1.0 - w0 - w1 - w3, 10)
        if w2 < -1e-9:
            continue
        w2 = max(w2, 0.0)
        w  = torch.tensor([w0, w1, w2, w3], dtype=torch.float32)
        # score = weighted sum over signals: (N, C, 4) @ (4,) → (N, C)
        scores = (signals * w.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        preds  = scores.argmax(dim=-1)
        acc    = (preds == labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_w   = (w0, w1, w2, w3)

    return torch.tensor(best_w, dtype=torch.float32), best_acc


def cross_entropy_loss(scores, labels):
    """Standard cross-entropy — directly maximises probability of correct image."""
    return torch.nn.functional.cross_entropy(scores, labels)


def combined_loss(scores, labels, margin=0.05, ce_weight=0.6, rank_weight=0.4):
    """
    Combined cross-entropy + pairwise ranking loss.
    CE directly optimises accuracy; ranking ensures correct > wrong by a margin.
    ce_weight=0.6, rank_weight=0.4 found to work well on small datasets.
    """
    ce   = cross_entropy_loss(scores, labels)
    rank = pairwise_ranking_loss(scores, labels, margin)
    return ce_weight * ce + rank_weight * rank


def train_task_gridsearch(task):
    """
    Grid search over signal weights — deterministic, no random init sensitivity.
    Finds optimal (w_ctx, w_fig, w_lit, w_gap) on training set.
    Saves weights as a lightweight 'model' dict.
    """
    print(f"\n{'='*60}")
    print(f"  Grid Search Phase 4  |  Task {task.upper()}")
    print(f"{'='*60}")

    train_cache = load_cache(task, "train")
    val_cache   = load_cache(task, "val")

    if train_cache is None:
        print(f"  [SKIP] No Phase 3 cache for Task {task.upper()}")
        return None

    tr_sig, tr_lbl = extract_4signals(train_cache, DEVICE)
    va_sig, va_lbl = extract_4signals(val_cache,   DEVICE) if val_cache else (None, None)

    print(f"  Train: {tr_sig.shape[0]} samples  |  Val: {va_sig.shape[0] if va_sig is not None else 0}")
    print(f"  Running grid search over (w_ctx, w_fig, w_lit, w_gap)...")

    best_w, train_acc = grid_search_weights(tr_sig, tr_lbl, steps=21)

    val_acc = 0.0
    if va_sig is not None:
        scores  = (va_sig * best_w.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        preds   = scores.argmax(dim=-1)
        val_acc = (preds == va_lbl).float().mean().item()

    print(f"  Best weights: ctx={best_w[0]:.2f}  fig={best_w[1]:.2f}  "
          f"lit={best_w[2]:.2f}  gap={best_w[3]:.2f}")
    print(f"  Train acc: {train_acc*100:.1f}%  |  Val acc: {val_acc*100:.1f}%")

    # Baseline: equal weights (0.25 each)
    eq_scores = tr_sig.mean(dim=-1)
    eq_preds  = eq_scores.argmax(-1)
    eq_acc    = (eq_preds == tr_lbl).float().mean().item()
    print(f"  Equal-weight baseline train: {eq_acc*100:.1f}%")

    # Save as a model checkpoint compatible with the Phase 4 loader
    # We save it as a tiny 4-signal LinearScorer with the found weights
    from phase4_model import Phase4CaptionFusedRanker
    model = Phase4CaptionFusedRanker(n_signals=4, head='linear').to(DEVICE)
    with torch.no_grad():
        model.head.scorer.weight.copy_(best_w.unsqueeze(0))
        model.head.scorer.bias.fill_(0.0)

    ckpt_path = os.path.join(MODELS_DIR, f"phase4_task_{task}_gridsearch.pt")
    torch.save({
        "epoch":       0,
        "model_state": model.state_dict(),
        "val_acc":     val_acc,
        "head":        "gridsearch",
        "weights":     best_w.tolist(),
    }, ckpt_path)
    print(f"  ✓ Saved: models/phase4_task_{task}_gridsearch.pt")
    return val_acc


def train_task(task, head='mlp', epochs=1000, lr=3e-3, patience=100,
               margin=0.05, weight_decay=1e-3):
    print(f"\n{'='*60}")
    print(f"  Training Phase 4  |  Task {task.upper()}  |  head={head}")
    print(f"{'='*60}")

    train_cache = load_cache(task, "train")
    val_cache   = load_cache(task, "val")

    if train_cache is None:
        print(f"  [SKIP] No Phase 3 cache found for Task {task.upper()} train.")
        return None

    # Use 4-signal extraction (drops noisy caption signals)
    train_signals, train_labels = extract_4signals(train_cache, DEVICE)
    print(f"  Train: {train_signals.shape[0]} samples, {train_signals.shape[1]} images each")
    print(f"  Signals: ctx→img, fig→img, lit→img, gap  (4 discriminative signals)")

    val_signals, val_labels = None, None
    if val_cache is not None:
        val_signals, val_labels = extract_4signals(val_cache, DEVICE)
        print(f"  Val:   {val_signals.shape[0]} samples")

    # ── Multi-start training: run N_RESTARTS times, keep best model ──────────
    N_RESTARTS  = 7
    best_overall_acc   = 0.0
    best_overall_state = None
    best_overall_epoch = 0

    for restart in range(N_RESTARTS):
        torch.manual_seed(restart * 137)

        model     = Phase4CaptionFusedRanker(n_signals=4, head=head).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                      patience=30, min_lr=1e-6)

        best_val_acc = 0.0
        best_state   = None
        best_epoch   = 0
        no_improve   = 0

        for epoch in range(1, epochs + 1):
            model.train()
            scores = model(train_signals)
            loss   = combined_loss(scores, train_labels, margin)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_acc = accuracy_from_signals(model, train_signals, train_labels)
            val_acc   = accuracy_from_signals(model, val_signals, val_labels) \
                        if val_signals is not None else 0.0

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch   = epoch
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        print(f"  Restart {restart+1}/{N_RESTARTS} — best val: {best_val_acc*100:.1f}%  (ep {best_epoch})")

        if best_val_acc > best_overall_acc:
            best_overall_acc   = best_val_acc
            best_overall_state = best_state
            best_overall_epoch = best_epoch

    # ── Save best model across all restarts ───────────────────────────────────
    if best_overall_state is not None:
        model.load_state_dict(best_overall_state)
        ckpt_path = os.path.join(MODELS_DIR, f"phase4_task_{task}_{head}.pt")
        torch.save({
            "epoch":       best_overall_epoch,
            "model_state": best_overall_state,
            "val_acc":     best_overall_acc,
            "head":        head,
        }, ckpt_path)

    model.count_parameters()
    print(f"\n  ✓ Best val accuracy: {best_overall_acc*100:.1f}%  (epoch {best_overall_epoch})")
    print(f"  ✓ Saved: models/phase4_task_{task}_{head}.pt")
    return best_overall_acc


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default="a", choices=["a", "b", "both"])
    parser.add_argument("--head",   default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--epochs", type=int,   default=1000)
    parser.add_argument("--lr",     type=float, default=3e-3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--margin",   type=float, default=0.05)
    args = parser.parse_args()

    tasks = ["a", "b"] if args.task == "both" else [args.task]

    gs_results  = {}
    mlp_results = {}

    for t in tasks:
        # Step 1: Grid search (deterministic, always runs first)
        gs_acc = train_task_gridsearch(t)
        if gs_acc is not None:
            gs_results[t] = gs_acc

        # Step 2: MLP training on 4 good signals (7 restarts)
        mlp_acc = train_task(
            task=t,
            head=args.head,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            margin=args.margin,
        )
        if mlp_acc is not None:
            mlp_results[t] = mlp_acc

    print(f"\n{'='*60}")
    print("  PHASE 4 TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<8} {'Grid Search':>12}  {'MLP (4-sig)':>12}  {'Best':>8}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}")
    for t in tasks:
        gs  = gs_results.get(t)
        mlp = mlp_results.get(t)
        best = max(v for v in [gs, mlp] if v is not None)
        gs_s  = f"{gs*100:.1f}%"  if gs  else "  —"
        mlp_s = f"{mlp*100:.1f}%" if mlp else "  —"
        print(f"  Task {t.upper():<3}  {gs_s:>12}  {mlp_s:>12}  {best*100:>7.1f}%")
    print(f"\n  Baseline:")
    print(f"    Phase 1 (CLIP zero-shot):  53.3%")
    print(f"    Phase 2 (IAPD + CN):       66.7%  ← previous best")
    print(f"    Phase 4 (IDIOM_DB prompts + optimised weights):  see above")

    print("""
  Baseline comparison:
    Phase 1 (CLIP zero-shot):  53.3%
    Phase 2 (IAPD + CN):       66.7%  ← previous best
    Phase 4 target:            > 66.7% (caption-fused)
""")
