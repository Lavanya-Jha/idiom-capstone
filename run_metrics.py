"""
run_metrics.py — Full Project Metrics Report
=============================================
Prints a complete evaluation summary across all phases.

HOW TO RUN:
    python run_metrics.py
"""

import os, sys, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import config_phase1 as cfg
from phase4_model import Phase4CaptionFusedRanker, extract_signals

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "cache")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEVICE     = "cpu"

def load_cache(task, split):
    path = os.path.join(CACHE_DIR, f"phase3_task_{task}_{split}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)

def accuracy(preds, labels):
    labels = labels.tolist() if hasattr(labels, "tolist") else labels
    preds  = preds.tolist()  if hasattr(preds,  "tolist") else preds
    return sum(p == l for p, l in zip(preds, labels)) / max(len(labels), 1)

def eval_cache(cache):
    iapd = cache["iapd_embeddings"].float()   # (N, 3, 512)
    imgs = cache["image_embeddings"].float()  # (N, C, 512)
    caps = cache["caption_embeddings"].float()
    gap  = cache["fig_lit_gap"].float()
    labels = cache["labels"]

    ctx = iapd[:, 2, :].unsqueeze(1)
    fig = iapd[:, 1, :].unsqueeze(1)
    lit = iapd[:, 0, :].unsqueeze(1)

    # Phase 1 proxy: literal text → image
    s_phase1 = (lit * imgs).sum(-1)
    # Phase 2 / baseline: contextual → image
    s_phase2 = (ctx * imgs).sum(-1)
    # Phase 3 a: caption rerank (figurative → caption)
    s_cap    = (fig * caps).sum(-1)
    # Phase 3 b: fig-lit gap
    s_gap    = gap
    # Phase 3 c: FLGS ensemble
    s_flgs   = 0.5 * s_phase2 + 0.3 * s_cap + 0.2 * s_gap

    results = {}
    for name, scores in [
        ("Phase 1  — CLIP literal zero-shot", s_phase1),
        ("Phase 2  — IAPD contextual (baseline)", s_phase2),
        ("Phase 3a — Caption rerank (fig→cap)", s_cap),
        ("Phase 3b — Figurative-Literal gap", s_gap),
        ("Phase 3c — FLGS ensemble", s_flgs),
    ]:
        preds = scores.argmax(-1)
        results[name] = accuracy(preds, labels)
    return results, labels

def eval_phase4_model(task, cache, labels, head='mlp'):
    ckpt_path = os.path.join(MODELS_DIR, f"phase4_task_{task}_{head}.pt")
    if not os.path.exists(ckpt_path):
        return None
    signals, _ = extract_signals(cache, DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = Phase4CaptionFusedRanker(n_signals=8, head=head).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        preds = model.predict(signals)
    return accuracy(preds, labels)

print()
print("=" * 65)
print("   FIGURATIVE LANGUAGE UNDERSTANDING — FULL METRICS REPORT")
print("=" * 65)

task_results = {}

for task in ["a", "b"]:
    task_label = f"Task {'A' if task == 'a' else 'B'}"
    val_c   = load_cache(task, "val")
    train_c = load_cache(task, "train")

    if val_c is None:
        print(f"\n  {task_label}: No cache found — skipping.")
        continue

    print(f"\n{'─'*65}")
    print(f"  {task_label.upper()}")
    print(f"{'─'*65}")
    n_train = len(train_c["labels"]) if train_c else 0
    n_val   = len(val_c["labels"])
    print(f"  Dataset  — Train: {n_train} idioms  |  Val: {n_val} idioms")
    print()

    val_res, val_labels = eval_cache(val_c)
    tr_res,  tr_labels  = eval_cache(train_c) if train_c else ({}, [])

    print(f"  {'Model / Scoring Strategy':<42} {'Val Acc':>8}  {'Train Acc':>10}")
    print(f"  {'─'*42}  {'─'*8}  {'─'*10}")

    for name, val_acc in val_res.items():
        tr_acc = tr_res.get(name, None)
        tr_str = f"{tr_acc*100:>9.1f}%" if tr_acc is not None else "       —"
        marker = ""
        print(f"  {name:<42} {val_acc*100:>7.1f}%  {tr_str}{marker}")

    # Phase 4
    for head in ["mlp", "linear"]:
        val_acc4 = eval_phase4_model(task, val_c, val_labels, head)
        if val_acc4 is not None:
            tr_acc4 = eval_phase4_model(task, train_c, tr_labels, head) if train_c else None
            tr_str  = f"{tr_acc4*100:>9.1f}%" if tr_acc4 is not None else "       —"
            baseline = val_res.get("Phase 2  — IAPD contextual (baseline)", 0)
            delta    = val_acc4 - baseline
            sign     = "+" if delta >= 0 else ""
            print(f"  Phase 4  — CaptionFused {head.upper():6s}         "
                  f"{val_acc4*100:>7.1f}%  {tr_str}   ({sign}{delta*100:.1f}% vs P2)")
        else:
            print(f"  Phase 4  — CaptionFused {head.upper():6s}             not trained")

    task_results[task] = val_res

# ── Summary Table ──────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  CROSS-PHASE ACCURACY SUMMARY")
print("=" * 65)
print(f"  {'Phase / Model':<42} {'Task A':>7}  {'Task B':>7}")
print(f"  {'─'*42}  {'─'*7}  {'─'*7}")

phases = [
    "Phase 1  — CLIP literal zero-shot",
    "Phase 2  — IAPD contextual (baseline)",
    "Phase 3c — FLGS ensemble",
]
for name in phases:
    a = task_results.get("a", {}).get(name)
    b = task_results.get("b", {}).get(name)
    a_str = f"{a*100:.1f}%" if a is not None else "   —"
    b_str = f"{b*100:.1f}%" if b is not None else "   —"
    print(f"  {name:<42} {a_str:>7}  {b_str:>7}")

# Phase 4
for task in ["a", "b"]:
    c = load_cache(task, "val")
    if c:
        v, lbl = eval_cache(c)
        task_results[task]["p4_val"] = eval_phase4_model(task, c, lbl) or 0

a4 = task_results.get("a", {}).get("p4_val")
b4 = task_results.get("b", {}).get("p4_val")
a4_s = f"{a4*100:.1f}%" if a4 else "   —"
b4_s = f"{b4*100:.1f}%" if b4 else "   —"
print(f"  {'Phase 4  — CaptionFused MLP (trained)':<42} {a4_s:>7}  {b4_s:>7}")
print()
print("  * Random baseline = 20.0% (1 in 5) for Task A")
print("  * Random baseline = 25.0% (1 in 4) for Task B")
print("=" * 65)
