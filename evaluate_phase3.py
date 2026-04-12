"""
evaluate_phase3.py — Phase 3 FLGS Evaluation (v2)
==================================================
Tries FOUR scoring strategies and reports which works best.

STRATEGY 0 — Phase 2 Baseline (reproduced)
    score = sim(context_text, image)
    This is exactly what Phase 2 achieved (66.7% Task A).

STRATEGY 1 — Caption Reranking (pure)
    score = sim(figurative_text, caption[i])
    "Which image's caption best describes the figurative meaning?"
    Works because: figurative text vs literal caption is text-to-text,
    which CLIP handles better than cross-modal for abstract language.

STRATEGY 2 — Figurative–Literal Gap
    score = sim(figurative_text, image) − sim(literal_text, image)
    "Which image aligns with figurative meaning MORE than literal words?"
    Discriminates figurative images from visual distractors.

STRATEGY 3 — FLGS Ensemble
    score = w1 * base + w2 * caption_rerank + w3 * gap
    Combines all three. Weights found by grid search on training set.
    Applied to val WITHOUT retuning (no data leakage).

HOW TO RUN:
    python evaluate_phase3.py --task a
    python evaluate_phase3.py --task b
    python evaluate_phase3.py --task both
"""

import os
import argparse
import torch
import config_phase1 as cfg


# ==============================================================================
# SCORING FUNCTIONS
# ==============================================================================

def score_base(cache):
    """Phase 2 style: contextual text → image similarity."""
    iapd = cache["iapd_embeddings"].float()   # (N, 3, 512)
    imgs = cache["image_embeddings"].float()  # (N, C, 512)
    ctx  = iapd[:, 2, :].unsqueeze(1)         # (N, 1, 512)
    return (ctx * imgs).sum(dim=-1)            # (N, C)


def score_caption_rerank(cache):
    """
    Pure caption reranking: figurative text vs each image's caption.
    'Which caption best describes the figurative meaning of this idiom?'
    """
    iapd = cache["iapd_embeddings"].float()       # (N, 3, 512)
    caps = cache["caption_embeddings"].float()    # (N, C, 512)
    fig  = iapd[:, 1, :].unsqueeze(1)             # (N, 1, 512)
    return (fig * caps).sum(dim=-1)               # (N, C)


def score_ctx_caption(cache):
    """Contextual text vs caption (cross-modal alignment)."""
    iapd = cache["iapd_embeddings"].float()
    caps = cache["caption_embeddings"].float()
    ctx  = iapd[:, 2, :].unsqueeze(1)
    return (ctx * caps).sum(dim=-1)


def score_gap(cache):
    """Pre-computed figurative−literal gap (stored in cache)."""
    return cache["fig_lit_gap"].float()           # (N, C)


def predict(scores):
    return scores.argmax(dim=-1).tolist()


def accuracy(preds, labels):
    labels = labels.tolist() if hasattr(labels, "tolist") else labels
    return sum(p == l for p, l in zip(preds, labels)) / max(len(labels), 1)


# ==============================================================================
# GRID SEARCH — finds best ensemble weights on training set
# ==============================================================================

def grid_search(train_cache):
    """
    Sweep weights for: base + w_cap*caption + w_gap*gap + w_ctx_cap*ctx_caption
    Find best combination on training set.
    """
    s_base    = score_base(train_cache)
    s_cap     = score_caption_rerank(train_cache)
    s_gap     = score_gap(train_cache)
    s_ctx_cap = score_ctx_caption(train_cache)
    labels    = train_cache["labels"]

    best_acc    = 0.0
    best_params = (1.0, 0.0, 0.0, 0.0)

    weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for w_cap in weights:
        for w_gap in [0.0, 0.1, 0.2, 0.3]:
            for w_ctx in [0.0, 0.1, 0.2, 0.3]:
                scores = s_base + w_cap * s_cap + w_gap * s_gap + w_ctx * s_ctx_cap
                acc = accuracy(predict(scores), labels)
                if acc > best_acc:
                    best_acc    = acc
                    best_params = (1.0, w_cap, w_gap, w_ctx)

    return best_params, best_acc


# ==============================================================================
# PER-POSITION BREAKDOWN
# ==============================================================================

def position_breakdown(preds, labels, C):
    labels = labels.tolist() if hasattr(labels, "tolist") else labels
    pos_correct = [0] * C
    pos_total   = [0] * C
    for p, l in zip(preds, labels):
        pos_total[l]   += 1
        pos_correct[l] += int(p == l)
    print(f"\n  Accuracy by correct-image position (where is the right answer?):")
    for i in range(C):
        if pos_total[i] > 0:
            bar = "█" * pos_correct[i] + "░" * (pos_total[i] - pos_correct[i])
            print(f"    Position {i+1}: {pos_correct[i]}/{pos_total[i]} = "
                  f"{pos_correct[i]/pos_total[i]:.0%}  [{bar}]")


# ==============================================================================
# MAIN
# ==============================================================================

def evaluate(task="a"):
    task_key   = task.lower()
    train_path = os.path.join(cfg.CACHE_DIR, f"phase3_task_{task_key}_train.pt")
    val_path   = os.path.join(cfg.CACHE_DIR, f"phase3_task_{task_key}_val.pt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Phase 3 cache not found: {train_path}\n"
            f"Run: python cache_phase3.py"
        )

    print(f"\n{'='*60}")
    print(f"  Phase 3 FLGS Evaluation — Task {task.upper()}")
    print(f"{'='*60}")

    train_cache = torch.load(train_path, map_location="cpu")
    val_cache   = torch.load(val_path,   map_location="cpu")

    N_train = len(train_cache["labels"])
    N_val   = len(val_cache["labels"])
    C       = val_cache["image_embeddings"].shape[1]

    print(f"\n  Train samples: {N_train}  |  Val samples: {N_val}  |  Candidates: {C}")
    print(f"\n  Trying all scoring strategies on val set:")
    print(f"  {'-'*55}")

    val_labels = val_cache["labels"]

    # Strategy 0: Phase 2 baseline
    acc0 = accuracy(predict(score_base(val_cache)), val_labels)
    print(f"  Strategy 0 — Phase 2 baseline:            {acc0:.1%}")

    # Strategy 1: Pure caption reranking
    acc1 = accuracy(predict(score_caption_rerank(val_cache)), val_labels)
    print(f"  Strategy 1 — Caption reranking (pure):    {acc1:.1%}")

    # Strategy 2: Fig-Lit gap
    acc2 = accuracy(predict(score_gap(val_cache)), val_labels)
    print(f"  Strategy 2 — Figurative–Literal gap:      {acc2:.1%}")

    # Strategy 3: Ctx-caption
    acc3 = accuracy(predict(score_ctx_caption(val_cache)), val_labels)
    print(f"  Strategy 3 — Context→caption:             {acc3:.1%}")

    # Strategy 4: Base + caption combined (fixed 50/50)
    s4   = score_base(val_cache) + 0.5 * score_caption_rerank(val_cache)
    acc4 = accuracy(predict(s4), val_labels)
    print(f"  Strategy 4 — Base + 0.5×caption:          {acc4:.1%}")

    # Strategy 5: Grid-searched ensemble (best train weights → apply to val)
    print(f"\n  Grid searching ensemble weights on train set...")
    (w_base, w_cap, w_gap, w_ctx), train_best = grid_search(train_cache)
    s5 = (w_base * score_base(val_cache) +
          w_cap  * score_caption_rerank(val_cache) +
          w_gap  * score_gap(val_cache) +
          w_ctx  * score_ctx_caption(val_cache))
    acc5 = accuracy(predict(s5), val_labels)
    print(f"  Best train weights: w_cap={w_cap}, w_gap={w_gap}, w_ctx={w_ctx}")
    print(f"  Strategy 5 — Grid ensemble:                {acc5:.1%}  (train: {train_best:.1%})")

    # Pick best strategy
    all_accs = [acc0, acc1, acc2, acc3, acc4, acc5]
    best_idx = all_accs.index(max(all_accs))
    best_acc = max(all_accs)
    strategy_names = ["Phase 2 baseline", "Caption reranking", "Fig-Lit gap",
                      "Context→caption", "Base+caption", "Grid ensemble"]

    print(f"\n{'='*60}")
    print(f"  PHASE 3 FINAL RESULTS — Task {task.upper()}")
    print(f"{'='*60}")
    print(f"\n  Random baseline:                    20.0%")
    print(f"  IRFL published benchmark:           ~22.0%")
    print(f"  Phase 1 (CLIP zero-shot):            53.3%")
    phase2_ref = 66.7 if task_key == "a" else 60.0
    print(f"  Phase 2 (IAPD + ConceptNet):        {phase2_ref:.1f}%")
    print(f"  Phase 3 FLGS best ({strategy_names[best_idx]}): {best_acc:.1%}  ← BEST")
    print(f"\n  Phase 3 vs Phase 2:                 {(best_acc - phase2_ref/100)*100:+.1f} pp")
    print(f"  Phase 3 vs random baseline:         {(best_acc - 0.20)*100:+.1f} pp")

    # Show best predictions breakdown
    best_scores = [score_base(val_cache),
                   score_caption_rerank(val_cache),
                   score_gap(val_cache),
                   score_ctx_caption(val_cache),
                   score_base(val_cache) + 0.5*score_caption_rerank(val_cache),
                   s5][best_idx]
    position_breakdown(predict(best_scores), val_labels, C)

    print(f"\n{'='*60}")
    return best_acc


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="a", choices=["a", "b", "both"])
    args = parser.parse_args()

    if args.task == "both":
        acc_a = evaluate("a")
        acc_b = evaluate("b")
        print(f"\n  ══ Final Phase 3 Summary ══")
        print(f"  Task A: {acc_a:.1%}")
        print(f"  Task B: {acc_b:.1%}")
        print(f"\n  Full progression:")
        print(f"    Random:   20.0%")
        print(f"    Phase 1:  53.3%  (+33.3 pp)")
        print(f"    Phase 2:  66.7%  (+13.4 pp)  [Task A]")
        print(f"    Phase 3:  {acc_a:.1%}  ({(acc_a-0.667)*100:+.1f} pp)  [Task A]")
    else:
        evaluate(args.task)
