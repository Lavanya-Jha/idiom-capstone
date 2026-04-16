"""
train_phase5.py
===============
PURPOSE:
    Phase 5 — YOLO-Enhanced Ranker.

    Extends Phase 4 by adding 2 YOLO-based signals to the 8 existing ones:

        per-image feature vector (10 scalars):
          [0] sim(ctx,   image)          — Phase 2 baseline
          [1] sim(fig,   image)          — figurative visual match
          [2] sim(lit,   image)          — literal visual match
          [3] sim(fig,   caption)        — figurative text vs image caption
          [4] sim(ctx,   caption)        — contextual text vs image caption
          [5] sim(lit,   caption)        — literal text vs image caption
          [6] sim(caption, image)        — caption ↔ image self-consistency
          [7] fig_lit_gap                — figurative−literal similarity gap
          [8] yolo_literal_score   [NEW] — YOLO detected objects vs literal idiom
          [9] yolo_scene_richness  [NEW] — object diversity in scene

    YOLO Signal Intuition:
        High yolo_literal_score → literal objects found in image → likely NOT the
        figurative interpretation → model should rank it lower.
        High yolo_scene_richness → complex multi-object scene → more likely figurative.

    MLPHead(10 → 16 → 1): ~204 trainable parameters.

HOW TO RUN:
    python train_phase5.py --task a
    python train_phase5.py --task b

REQUIREMENTS:
    - cache/phase3_task_*.pt  (from cache_phase3.py)
    - cache/yolo_task_*.pt    (from cache_yolo.py)

OUTPUT:
    models/phase5_task_a_mlp.pt
    models/phase5_task_b_mlp.pt
"""

import os
import argparse
import torch
import torch.nn.functional as F

from phase4_model import Phase4CaptionFusedRanker, pairwise_ranking_loss, extract_signals
import config_phase1 as cfg

# Use script's own directory for cache/model paths (works from any location)
_HERE     = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "cache")
MODEL_DIR = os.path.join(_HERE, "models")


# ==============================================================================
# YOLO SIGNAL EXTRACTION
# ==============================================================================

def extract_yolo_signals(phase3_cache, yolo_cache, device="cpu"):
    """
    Combine Phase 3 signals (8) with YOLO signals (2) → 10 total.

    Handles size mismatch: Phase 3 may have more samples than YOLO cache
    (e.g. Phase 3 includes Extended set). Aligns by idiom name — idioms
    missing from YOLO cache get 0.0 for both YOLO signals.

    Returns:
        signals: (N, C, 10) — 10 scalar features per image
        labels:  (N,)
    """
    signals_8, labels = extract_signals(phase3_cache, device)  # (N, C, 8)

    N, C, _ = signals_8.shape
    p3_idioms   = phase3_cache["idioms"]          # list[str], length N
    yolo_idioms = yolo_cache["idioms"]            # list[str], length M (may differ)
    yolo_lit_m  = yolo_cache["yolo_literal_score"].float()    # (M, C)
    yolo_rich_m = yolo_cache["yolo_scene_richness"].float()   # (M, C)

    # Build lookup: idiom string → row index in YOLO cache
    yolo_idx = {idiom: i for i, idiom in enumerate(yolo_idioms)}

    # Align YOLO signals to Phase 3 order; 0.0 for any idiom not in YOLO cache
    lit_aligned  = torch.zeros(N, C)
    rich_aligned = torch.zeros(N, C)
    matched = 0
    for i, idiom in enumerate(p3_idioms):
        if idiom in yolo_idx:
            j = yolo_idx[idiom]
            lit_aligned[i]  = yolo_lit_m[j]
            rich_aligned[i] = yolo_rich_m[j]
            matched += 1

    print(f"  YOLO alignment: {matched}/{N} Phase3 idioms matched in YOLO cache")
    if matched < N:
        print(f"  ({N - matched} idioms from Extended set — YOLO signals set to 0.0 for these)")

    lit_aligned  = lit_aligned.to(device)
    rich_aligned = rich_aligned.to(device)

    signals_10 = torch.cat([
        signals_8,
        lit_aligned.unsqueeze(-1),   # (N, C, 1)
        rich_aligned.unsqueeze(-1),  # (N, C, 1)
    ], dim=-1)

    return signals_10, labels


# ==============================================================================
# TRAINING
# ==============================================================================

def train(task="a", epochs=500, lr=1e-3, weight_decay=1e-3, patience=60, margin=0.05):
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = "cpu"

    # ── Load caches ──────────────────────────────────────────────────────────
    p3_train_path   = os.path.join(CACHE_DIR, f"phase3_task_{task}_train.pt")
    p3_val_path     = os.path.join(CACHE_DIR, f"phase3_task_{task}_val.pt")
    yolo_train_path = os.path.join(CACHE_DIR, f"yolo_task_{task}_train.pt")
    yolo_val_path   = os.path.join(CACHE_DIR, f"yolo_task_{task}_val.pt")

    for path in [p3_train_path, p3_val_path, yolo_train_path, yolo_val_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache not found: {path}\n"
                f"Run cache_phase3.py then cache_yolo.py first."
            )

    print(f"Loading Phase 3 caches...")
    p3_train   = torch.load(p3_train_path,   map_location=device)
    p3_val     = torch.load(p3_val_path,     map_location=device)

    print(f"Loading YOLO caches...")
    yolo_train = torch.load(yolo_train_path, map_location=device)
    yolo_val   = torch.load(yolo_val_path,   map_location=device)

    # ── Extract combined signals ──────────────────────────────────────────────
    train_signals, train_labels = extract_yolo_signals(p3_train, yolo_train, device)
    val_signals,   val_labels   = extract_yolo_signals(p3_val,   yolo_val,   device)

    N_train = train_signals.shape[0]
    N_val   = val_signals.shape[0]
    N_sig   = train_signals.shape[2]  # 10

    print(f"\n  Task {task.upper()} — Phase 5 YOLO Training")
    print(f"  Train: {N_train} samples | Val: {N_val} samples | Signals: {N_sig}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Phase4CaptionFusedRanker(n_signals=N_sig, head="mlp", hidden=16, dropout=0.1)
    model.count_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20
    )

    best_val_acc  = 0.0
    best_epoch    = 0
    no_improve    = 0
    model_path    = os.path.join(MODEL_DIR, f"phase5_task_{task}_mlp.pt")

    print(f"\n  Training for up to {epochs} epochs (patience={patience})...\n")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Acc':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*9}  {'-'*8}")

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        scores = model(train_signals)                         # (N, C)
        loss   = pairwise_ranking_loss(scores, train_labels, margin=margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Evaluate ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            train_preds = model(train_signals).argmax(dim=-1)  # (N,)
            val_preds   = model(val_signals).argmax(dim=-1)    # (N,)

        train_acc = (train_preds == train_labels).float().mean().item()
        val_acc   = (val_preds   == val_labels  ).float().mean().item()

        scheduler.step(val_acc)

        if epoch % 25 == 0 or epoch == 1:
            print(f"  {epoch:>5}  {loss.item():>10.4f}  {train_acc:>9.1%}  {val_acc:>8.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\n  Best val accuracy: {best_val_acc:.1%} (epoch {best_epoch})")
    print(f"  Model saved to: {model_path}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        final_preds = model(val_signals).argmax(dim=-1)
    final_acc = (final_preds == val_labels).float().mean().item()
    print(f"\n  Final val accuracy (best checkpoint): {final_acc:.1%}")

    # Compare with Phase 4 baseline (8-signal)
    print("\n  Signal contribution summary:")
    print(f"    Phase 4 (8 signals): see models/phase4_task_{task}_mlp.pt")
    print(f"    Phase 5 (10 signals, +YOLO): {final_acc:.1%}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["a", "b"], default="a")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=60)
    args = parser.parse_args()

    train(
        task=args.task,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )
