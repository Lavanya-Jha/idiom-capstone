"""
phase4_model.py
===============
PURPOSE:
    Phase 4 — Caption-Fused Ranker, adapted from the Subtask B MLPHead
    architecture described in the companion codebase (idiom_classifier/).

    The Subtask B model concatenated:
        [CLIP_text_embed | CLIP_image_embed | cosine_sim]  →  MLPHead  →  score

    For Subtask A we extend this with caption embeddings (available in our
    Phase 3 cache) and operate exclusively on SCALAR cosine similarities to
    avoid overfitting on the ~100-sample dataset:

        per-image feature vector (8 scalars):
          [0] sim(ctx,   image)     — Phase 2 baseline signal
          [1] sim(fig,   image)     — figurative visual match
          [2] sim(lit,   image)     — literal visual match
          [3] sim(fig,   caption)   — figurative text vs image caption  ← KEY NEW
          [4] sim(ctx,   caption)   — contextual text vs image caption  ← NEW
          [5] sim(lit,   caption)   — literal text vs image caption     ← NEW
          [6] sim(caption, image)   — caption ↔ image self-consistency  ← NEW
          [7] fig_lit_gap           — pre-computed gap (from Phase 3)

    Why scalar signals?
        With only ~100 training samples, using raw 512-dim embeddings as inputs
        (as Subtask B does with 1025-dim vectors) risks memorisation.  Working
        with 8 scalar similarities means the model can learn robust reweighting
        without overfitting.

    MLPHead (adapted from Subtask B):
        Linear(8 → 16) + LayerNorm + GELU + Dropout
        Linear(16 → 1)
        + residual shortcut: Linear(8 → 1) added to output
        Total trainable parameters: 8*16+16 + 16*1+1 + 8*1+1 = 170 params

    For a tiny dataset, 170 params is still generous.  We also support a
    "linear-only" mode (10 params) for maximum regularisation.

TRAINING:
    Pairwise MarginRankingLoss — adapted from the Subtask B suggestion.
    For each sample, the correct image (label) should score higher than all
    wrong images by at least `margin`.  This is more aligned with the ranking
    task than CrossEntropyLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# SIGNAL EXTRACTION
# ==============================================================================

def extract_signals(cache, device="cpu"):
    """
    From Phase 3 cache, extract 8 scalar signals per image per sample.

    Args:
        cache: dict from torch.load(phase3_*.pt)
        device: torch device

    Returns:
        signals: (N, C, 8) tensor  — 8 scalar features per image
        labels:  (N,)     tensor  — index of correct image
    """
    iapd = cache["iapd_embeddings"].float().to(device)   # (N, 3, 512)
    imgs = cache["image_embeddings"].float().to(device)  # (N, C, 512)
    caps = cache["caption_embeddings"].float().to(device)# (N, C, 512)
    gap  = cache["fig_lit_gap"].float().to(device)       # (N, C)
    lbls = cache["labels"].long().to(device)             # (N,)

    N, C, D = imgs.shape

    lit = iapd[:, 0, :].unsqueeze(1)   # (N, 1, 512)
    fig = iapd[:, 1, :].unsqueeze(1)   # (N, 1, 512)
    ctx = iapd[:, 2, :].unsqueeze(1)   # (N, 1, 512)

    # Each sim: (N, C)
    s_ctx_img = (ctx * imgs).sum(dim=-1)          # [0]
    s_fig_img = (fig * imgs).sum(dim=-1)          # [1]
    s_lit_img = (lit * imgs).sum(dim=-1)          # [2]
    s_fig_cap = (fig * caps).sum(dim=-1)          # [3]  ← KEY NEW
    s_ctx_cap = (ctx * caps).sum(dim=-1)          # [4]
    s_lit_cap = (lit * caps).sum(dim=-1)          # [5]
    s_cap_img = (caps * imgs).sum(dim=-1)         # [6]
    # gap                                          # [7]

    signals = torch.stack([
        s_ctx_img, s_fig_img, s_lit_img,
        s_fig_cap, s_ctx_cap, s_lit_cap,
        s_cap_img, gap
    ], dim=-1)  # (N, C, 8)

    return signals, lbls


# ==============================================================================
# MLPHEAD ADAPTED FROM SUBTASK B
# ==============================================================================

class MLPHead(nn.Module):
    """
    Small MLP scorer adapted from the Subtask B IdiomModel's MLP head.

    Subtask B used:  Linear(1025→512) → LN → GELU → Dropout → Linear(512→256) → LN → GELU → Dropout → Linear(256→1)
    Our version:     Linear(8→16)     → LN → GELU → Dropout → Linear(16→1)
                     + shortcut Linear(8→1)   (residual connection)

    Total trainable: 8*16+16 + 16+16 + 16*1+1 + 8*1+1 = 170 params
    """
    def __init__(self, n_signals=8, hidden=16, dropout=0.1):
        super().__init__()
        self.fc1      = nn.Linear(n_signals, hidden)
        self.ln1      = nn.LayerNorm(hidden)
        self.drop     = nn.Dropout(dropout)
        self.fc2      = nn.Linear(hidden, 1)
        self.shortcut = nn.Linear(n_signals, 1, bias=False)  # residual

        # Init shortcut to equal weights (prior: all signals equally useful)
        nn.init.constant_(self.shortcut.weight, 1.0 / n_signals)

    def forward(self, x):
        """
        Args:
            x: (..., n_signals)
        Returns:
            score: (..., 1)
        """
        out = F.gelu(self.ln1(self.fc1(x)))
        out = self.drop(out)
        out = self.fc2(out)
        out = out + self.shortcut(x)   # residual
        return out


class LinearScorer(nn.Module):
    """
    Ultra-minimal baseline: learned linear combination of 8 signals.
    10 trainable parameters.  Use when MLPHead overfits.
    """
    def __init__(self, n_signals=8):
        super().__init__()
        self.scorer = nn.Linear(n_signals, 1)
        nn.init.constant_(self.scorer.weight, 1.0 / n_signals)
        nn.init.constant_(self.scorer.bias, 0.0)

    def forward(self, x):
        return self.scorer(x)


# ==============================================================================
# FULL PHASE 4 MODEL
# ==============================================================================

class Phase4CaptionFusedRanker(nn.Module):
    """
    Complete Phase 4 model.

    Input:  Phase 3 cache signals tensor (N, C, 8)
    Output: scores (N, C) — higher = more likely to be the figurative image

    Architecture options:
        'mlp'    : MLPHead (170 params)  ← recommended
        'linear' : LinearScorer (10 params) ← maximum regularisation

    Training objective:
        Pairwise MarginRankingLoss — the correct image should score higher
        than every wrong image by at least `margin`.
    """
    def __init__(self, n_signals=8, head='mlp', hidden=16, dropout=0.1):
        super().__init__()
        self.n_signals = n_signals

        if head == 'mlp':
            self.head = MLPHead(n_signals, hidden, dropout)
        elif head == 'linear':
            self.head = LinearScorer(n_signals)
        else:
            raise ValueError(f"Unknown head type: {head}. Use 'mlp' or 'linear'.")

    def forward(self, signals):
        """
        Args:
            signals: (B, C, n_signals) or (B, n_signals) for single-image

        Returns:
            scores: (B, C) or (B,)
        """
        squeeze = signals.dim() == 2
        if squeeze:
            signals = signals.unsqueeze(1)  # (B, 1, n_signals)

        B, C, _ = signals.shape
        flat  = signals.view(B * C, self.n_signals)   # (B*C, n_signals)
        score = self.head(flat).view(B, C)             # (B, C)

        if squeeze:
            score = score.squeeze(1)
        return score

    def predict(self, signals):
        with torch.no_grad():
            scores = self.forward(signals)
            return scores.argmax(dim=-1)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  Phase 4 Model Parameters:")
        for name, mod in self.named_children():
            n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            print(f"    {name:20s}: {n:>6,}")
        print(f"    {'TOTAL TRAINABLE':20s}: {total:>6,}")
        return total


# ==============================================================================
# PAIRWISE RANKING LOSS  (adapted from Subtask B suggestion)
# ==============================================================================

def pairwise_ranking_loss(scores, labels, margin=0.05):
    """
    MarginRankingLoss adapted for multi-image ranking.

    For each sample, the correct image (given by label) should score higher
    than every wrong image by at least `margin`.

    Args:
        scores: (B, C) raw scores per image
        labels: (B,)  index of correct (figurative) image
        margin: minimum score gap required

    Returns:
        loss: scalar
    """
    B, C = scores.shape
    loss = torch.tensor(0.0, device=scores.device, requires_grad=True)

    for b in range(B):
        correct_score = scores[b, labels[b]]   # scalar
        loss_b = torch.tensor(0.0, device=scores.device)
        count = 0
        for c in range(C):
            if c == labels[b]:
                continue
            wrong_score = scores[b, c]
            # correct should be > wrong by margin
            pair_loss = F.relu(margin - (correct_score - wrong_score))
            loss_b = loss_b + pair_loss
            count += 1
        if count > 0:
            loss = loss + loss_b / count

    return loss / B
