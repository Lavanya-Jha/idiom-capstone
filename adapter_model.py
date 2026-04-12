"""
adapter_model.py
================
PURPOSE:
    Defines the lightweight FigurativeAdapter model that is trained on top of
    completely frozen CLIP embeddings.

WHY AN ADAPTER AND NOT FINE-TUNING CLIP?
─────────────────────────────────────────
    CLIP has 151 million parameters trained on 400 million image-text pairs.
    We have ~60 training samples.

    If we fine-tune CLIP on 60 samples:
        → The model MEMORIZES training data (overfitting)
        → CLIP's powerful pre-trained knowledge is DESTROYED (catastrophic forgetting)
        → Validation accuracy collapses

    The adapter approach:
        → CLIP is completely frozen (we never touch its weights)
        → We train ONLY a tiny 2-layer MLP with ~132K parameters
        → The adapter SHIFTS CLIP's embeddings slightly toward figurative meaning
        → Because it's so small, it cannot overfit on 60 samples
        → CLIP's knowledge is fully preserved

ARCHITECTURE DIAGRAM:
─────────────────────
    [Cached CLIP Text Embedding: 512-dim]
                ↓
        ┌───────────────────┐
        │   Text Adapter    │  (512 → 128 → 512, residual)
        └───────────────────┘
                ↓
    [Adapted Text Embedding: 512-dim, normalized]
                ↓
    ─────── cosine similarity / temperature ───────
                ↑
    [Adapted Image Embeddings: (num_cands, 512-dim)]
                ↑
        ┌───────────────────┐
        │   Image Adapter   │  (512 → 128 → 512, residual)
        └───────────────────┘
                ↑
    [Cached CLIP Image Embeddings: (num_cands, 512-dim)]

    Output: Similarity scores (num_cands,) → softmax → ranking

TRAINABLE PARAMETERS:
─────────────────────
    Text Adapter:
        down_proj:   512 × 128 = 65,536
        down_bias:   128
        up_proj:     128 × 512 = 65,536
        up_bias:     512
        layer_norm:  512 × 2   = 1,024
        alpha:       1
        subtotal:    ~132,737

    Image Adapter:    same as text = ~132,737

    Temperature:      1

    TOTAL:            ~265,475 parameters  (vs CLIP's 151,000,000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterMLP(nn.Module):
    """
    Single bottleneck adapter block with residual connection.

    Structure:  x → LayerNorm → Linear(512→128) → GELU → Dropout
                  → Linear(128→512) → scale by alpha → add residual x → normalize

    The residual connection is KEY:
        output = normalize(x + alpha * f(x))

    At initialization, alpha = 0.1 (small), so output ≈ x (CLIP embeddings).
    The adapter starts as near-identity and gradually learns what to change.
    This is much more stable than replacing CLIP features entirely.
    """

    def __init__(self, embed_dim=512, bottleneck_dim=128, dropout=0.2):
        super().__init__()

        self.down_proj  = nn.Linear(embed_dim,      bottleneck_dim)  # 512 → 128
        self.up_proj    = nn.Linear(bottleneck_dim, embed_dim)       # 128 → 512
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Alpha: how much does the adapter contribution matter?
        # Learned during training, starts small so we don't disrupt CLIP features immediately
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Initialize weights to near-zero — adapter starts as identity transform
        # This is important! Random init would immediately destroy CLIP embeddings.
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight,   gain=0.01)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """
        Args:
            x: (batch_size, embed_dim) — input CLIP embeddings, already L2-normalized

        Returns:
            out: (batch_size, embed_dim) — adapted embeddings, L2-normalized
        """
        residual = x                             # Save for skip connection

        h = self.down_proj(x)                   # (B, 128) — compress
        h = F.gelu(h)                           # Non-linearity (smoother than ReLU)
        h = self.dropout(h)                     # Regularization
        h = self.up_proj(h)                     # (B, 512) — expand back

        # Residual: mostly original CLIP features + small adapter contribution
        out = self.layer_norm(residual + self.alpha * h)

        # Re-normalize: CLIP embeddings are always unit vectors
        # Cosine similarity requires this
        out = F.normalize(out, dim=-1)

        return out


class FigurativeAdapter(nn.Module):
    """
    Full adapter model: one AdapterMLP for text, one for images.

    This is the ENTIRE trainable model for Phase 1.
    CLIP is not part of this class — it's already been run offline.

    Input:  Pre-cached frozen CLIP embeddings (loaded from cache/*.pt)
    Output: Similarity scores for each candidate image
    """

    def __init__(self, embed_dim=512, bottleneck_dim=128, temperature=0.07):
        super().__init__()

        self.text_adapter  = AdapterMLP(embed_dim, bottleneck_dim)
        self.image_adapter = AdapterMLP(embed_dim, bottleneck_dim)

        # Learnable temperature parameter
        # We store log(temperature) so it's always positive after exp()
        # Lower temperature = sharper softmax = model is more "decisive"
        # Higher temperature = flatter softmax = model is more "uncertain"
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    @property
    def temperature(self):
        """Returns temperature clamped to a safe range."""
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def forward(self, text_embeds, image_embeds):
        """
        Compute similarity scores between adapted text and image embeddings.

        Args:
            text_embeds:  (batch_size, 512)
                          One text embedding per idiom (already L2-normalized)

            image_embeds: (batch_size, num_candidates, 512)
                          Multiple image embeddings per idiom (already L2-normalized)

        Returns:
            scores: (batch_size, num_candidates)
                    Raw (unormalized) similarity scores — higher = better match
                    These are passed to cross-entropy loss or softmax for ranking
        """
        B, N, D = image_embeds.shape          # batch, num_candidates, embed_dim

        # ── Adapt text embeddings ─────────────────────────────────────────────
        adapted_text = self.text_adapter(text_embeds)                    # (B, 512)

        # ── Adapt image embeddings ────────────────────────────────────────────
        # Reshape to (B*N, 512) so we can run the adapter in one pass
        adapted_images = self.image_adapter(
            image_embeds.view(B * N, D)                                  # (B*N, 512)
        ).view(B, N, D)                                                  # (B, N, 512)

        # ── Compute cosine similarity ─────────────────────────────────────────
        # Expand text from (B, 512) to (B, N, 512) to match image shape
        adapted_text_exp = adapted_text.unsqueeze(1).expand(B, N, D)    # (B, N, 512)

        # Element-wise multiply then sum = dot product of normalized vectors = cosine sim
        scores = (adapted_text_exp * adapted_images).sum(dim=-1)         # (B, N)

        # Scale by temperature: dividing by small temperature amplifies differences
        scores = scores / self.temperature

        return scores

    def predict(self, text_embeds, image_embeds):
        """
        Get predicted class (best image index) for evaluation.

        Args:
            text_embeds:  (batch_size, 512)
            image_embeds: (batch_size, num_candidates, 512)

        Returns:
            predictions: (batch_size,) — index of best matching image
        """
        with torch.no_grad():
            scores = self.forward(text_embeds, image_embeds)             # (B, N)
            predictions = scores.argmax(dim=-1)                          # (B,)
        return predictions

    def count_parameters(self):
        """Print a summary of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  Model Parameter Count:")
        print(f"    Text Adapter:   {sum(p.numel() for p in self.text_adapter.parameters()):>10,}")
        print(f"    Image Adapter:  {sum(p.numel() for p in self.image_adapter.parameters()):>10,}")
        print(f"    Temperature:    {1:>10,}")
        print(f"    TOTAL:          {total:>10,}")
        print(f"    (CLIP has 151,000,000 — we train {total/151_000_000*100:.2f}% of that)")
        return total
