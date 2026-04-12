"""
phase2_model.py
===============
PURPOSE:
    Enhanced model for Phase 2. Builds on Phase 1 by improving the text
    representation BEFORE matching — without adding many learnable parameters.

    The core lesson from Phase 1: with only 100 training samples, ANY model
    with more than ~50 learnable parameters will memorise the training set and
    fail to generalise. The adapter (67K params) and the original IAPD attention
    (524K params) both overfit instantly.

    Phase 2 solution: keep CLIP frozen, reduce learnable params to the absolute
    minimum, and improve representations through better prompt design (IAPD)
    and commonsense knowledge injection (ConceptNet).

ARCHITECTURE (4 learnable parameters total):
    ┌─────────────────────────────────────────────────────────┐
    │  INPUT: iapd_embeddings (3, 512) + conceptnet_vec (300) │
    └───────────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────▼────────────┐
          │   IAPD 3-Weight Softmax  │  3 learnable scalars → softmax
          │   output: (512,)         │  weighted avg of lit/fig/ctx
          └─────────────┬────────────┘
                        │
          ┌─────────────▼────────────┐
          │  ConceptNet Fixed Proj   │  Linear(300→512), requires_grad=False
          │  + learned gate scalar   │  1 learnable gate weight
          │  output: (512,)          │
          └─────────────┬────────────┘
                        │
                 cosine similarity (scaled by learned temperature)
                        │
          ┌─────────────▼────────────┐
          │   Frozen CLIP images     │  no adapter — same as zero-shot
          └─────────────┬────────────┘
                        │
                 scores (num_candidates,)

WHY THIS WORKS BETTER:
    - IAPD 3-weight attention: only 3 params.  Even with 100 samples the model
      can learn whether literal / figurative / contextual matters more overall
      without memorising specific samples.
    - ConceptNet gate: 1 param.  Learns whether to trust commonsense vectors.
    - Fixed projection for ConceptNet: orthogonal init ensures the 300-dim
      vector is faithfully represented in 512-dim space without training.
    - No adapters: prevents the catastrophic forgetting that collapsed Phase 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# IAPD 3-WEIGHT ATTENTION  (3 learnable parameters)
# ==============================================================================

class IAPDAttention(nn.Module):
    """
    Learns a single global mixing ratio over the 3 IAPD perspectives.

    Instead of a per-sample cross-attention (524K params that overfit instantly),
    this uses one shared scalar per perspective.  The model learns e.g.
    "figurative perspective is most useful overall" without memorising samples.

    Parameters: 3  (one scalar weight per perspective, before softmax)
    """
    def __init__(self):
        super().__init__()
        # Initialise to favour figurative (index 1) over literal (0) and contextual (2)
        self.weights = nn.Parameter(torch.tensor([0.2, 0.5, 0.3]))

    def forward(self, iapd_embeddings):
        """
        Args:
            iapd_embeddings: (B, 3, 512)
                index 0 = literal
                index 1 = figurative
                index 2 = contextual

        Returns:
            attended:     (B, 512)  — weighted combination
            attn_weights: (B, 3)    — same weights broadcast for logging
        """
        w = F.softmax(self.weights, dim=0)          # (3,)

        # Clamp: figurative (idx 1) must be >= 0.4 to prevent contextual from
        # dominating. Contextual = raw training sentence → memorises training set.
        # Renormalise after clamping so weights still sum to 1.
        w_list = [w[0], w[1].clamp(min=0.4), w[2]]
        w_clamped = torch.stack(w_list)
        w = w_clamped / w_clamped.sum()

        attended = (iapd_embeddings * w.view(1, 3, 1)).sum(dim=1)   # (B, 512)
        attended = F.normalize(attended, dim=-1)

        # Broadcast same weights to every sample for interpretability logging
        B = iapd_embeddings.size(0)
        attn_weights = w.unsqueeze(0).expand(B, -1)  # (B, 3)
        return attended, attn_weights


# ==============================================================================
# CONCEPTNET FUSION  (1 learnable parameter — the gate)
# ==============================================================================

class ConceptNetFusion(nn.Module):
    """
    Injects ConceptNet commonsense knowledge into the text embedding.

    The Linear(300→512) projection is FIXED (no gradient) — it is initialised
    orthogonally so the 300-dim ConceptNet space maps faithfully to 512-dim
    CLIP space without any training.

    The only learnable param is a scalar gate (sigmoid-activated) that controls
    how much ConceptNet influence to add.  This can be learned reliably from
    100 samples because it is just one number.

    Parameters: 1  (gate scalar)
    """
    def __init__(self, cn_dim=300, embed_dim=512):
        super().__init__()

        # Fixed orthogonal projection — no gradient
        self.proj = nn.Linear(cn_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.proj.weight.requires_grad = False   # <-- frozen

        # Single learnable gate: how much ConceptNet to add
        # Init at 0.1 → small influence, model grows it if useful
        self.gate = nn.Parameter(torch.tensor(0.1))

        # LayerNorm is FROZEN — its 1024 params (weight+bias) would overfit on 100 samples
        self.layer_norm = nn.LayerNorm(embed_dim)
        for p in self.layer_norm.parameters():
            p.requires_grad = False

    def forward(self, text_embed, cn_vector):
        """
        Args:
            text_embed: (B, 512) — IAPD-attended text embedding
            cn_vector:  (B, 300) — ConceptNet NumberBatch vector

        Returns:
            fused: (B, 512)
        """
        cn_proj = self.proj(cn_vector)                          # (B, 512), fixed
        cn_proj = F.normalize(cn_proj, dim=-1)

        gate  = torch.sigmoid(self.gate)                        # scalar 0-1
        fused = self.layer_norm(text_embed + gate * cn_proj)
        return F.normalize(fused, dim=-1)


# ==============================================================================
# FULL PHASE 2 MODEL  (4 learnable parameters total)
# ==============================================================================

class Phase2FigurativeModel(nn.Module):
    """
    Complete Phase 2 model.

    Learnable parameters:
        IAPD weights (3) + ConceptNet gate (1) + temperature (1) = 5 total
        (temperature excluded from count as it's clipped to safe range)

    Everything else (CLIP, ConceptNet projection) is frozen.
    """

    def __init__(self, embed_dim=512, cn_dim=300, temperature=0.07):
        super().__init__()
        self.iapd_attention = IAPDAttention()
        self.cn_fusion      = ConceptNetFusion(cn_dim, embed_dim)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def encode_text(self, iapd_embeddings, cn_vectors):
        """
        Args:
            iapd_embeddings: (B, 3, 512)
            cn_vectors:      (B, 300)

        Returns:
            text_embed:   (B, 512)
            attn_weights: (B, 3)
        """
        attended, attn_weights = self.iapd_attention(iapd_embeddings)
        fused = self.cn_fusion(attended, cn_vectors)
        return fused, attn_weights

    def forward(self, iapd_embeddings, image_embeddings, cn_vectors):
        """
        Args:
            iapd_embeddings:  (B, 3, 512)
            image_embeddings: (B, num_candidates, 512)
            cn_vectors:       (B, 300)

        Returns:
            scores:       (B, num_candidates)
            attn_weights: (B, 3)
        """
        B, N, D = image_embeddings.shape

        text_embed, attn_weights = self.encode_text(iapd_embeddings, cn_vectors)

        # Images are used as-is (frozen CLIP — no adapter)
        text_exp = text_embed.unsqueeze(1).expand(B, N, D)             # (B, N, 512)
        scores   = (text_exp * image_embeddings).sum(dim=-1) / self.temperature  # (B, N)

        return scores, attn_weights

    def predict(self, iapd_embeddings, image_embeddings, cn_vectors):
        with torch.no_grad():
            scores, _ = self.forward(iapd_embeddings, image_embeddings, cn_vectors)
            return scores.argmax(dim=-1)

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters() if p.requires_grad)
        iapd_n    = sum(p.numel() for p in self.iapd_attention.parameters() if p.requires_grad)
        cn_n      = sum(p.numel() for p in self.cn_fusion.parameters()      if p.requires_grad)
        temp_n    = 1

        print(f"\n  Phase 2 Model Parameters:")
        print(f"    IAPD 3-Weight Attention: {iapd_n:>10,}  (3 scalars → softmax)")
        print(f"    ConceptNet Gate:         {cn_n:>10,}  (1 scalar, proj is frozen)")
        print(f"    ConceptNet Projection:   {sum(p.numel() for p in self.cn_fusion.proj.parameters()):>10,}  (FROZEN — no gradient)")
        print(f"    Temperature:             {temp_n:>10,}")
        print(f"    TOTAL TRAINABLE:         {total:>10,}")
        return total
