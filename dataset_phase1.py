"""
dataset_phase1.py
=================
PURPOSE:
    PyTorch Dataset that loads from PRE-COMPUTED CLIP embeddings (cache/*.pt).

WHY NOT LOAD IMAGES DIRECTLY?
    Loading images → running CLIP → getting embeddings happens in cache_embeddings.py
    After that, this Dataset simply reads tensors from disk.

    Each __getitem__ call takes ~0.1ms instead of ~2000ms.
    This is why training goes from 30+ minutes/epoch to 1-2 minutes/epoch.

AUGMENTATION (Embedding-Space):
    We can't do traditional image augmentation (flips, crops) on cached embeddings.
    Instead we add tiny Gaussian noise in embedding space.

    Why this helps:
        - CLIP embeddings are smooth: nearby points in embedding space
          are semantically similar
        - Adding noise ε ~ N(0, 0.01) creates slightly different versions
          of each sample without changing semantics
        - This effectively multiplies training data 2-3x (augment=True during training)
        - Never use augmentation during validation — we want deterministic eval
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CachedEmbeddingDataset(Dataset):
    """
    Loads pre-cached CLIP embeddings for the figurative language task.

    Each sample contains:
        text_embed:   (512,)                  — CLIP text embedding for the idiom
        image_embeds: (num_candidates, 512)   — CLIP embeddings for all candidate images
        label:        int                     — index of the correct image (0-based)
        idiom:        str                     — the idiom string (for logging)
    """

    def __init__(self, cache_path, augment=False, noise_scale=0.01):
        """
        Args:
            cache_path:   Path to .pt file created by cache_embeddings.py
            augment:      If True, add Gaussian noise (use for training only)
            noise_scale:  Standard deviation of noise (0.01 is safe, keep small)
        """
        print(f"\nLoading dataset from: {cache_path}")

        data = torch.load(cache_path, map_location='cpu')

        self.text_embeddings  = data['text_embeddings'].float()    # (N, 512)
        self.image_embeddings = data['image_embeddings'].float()   # (N, num_cands, 512)
        self.labels           = data['labels']                      # (N,)
        self.idioms           = data['idioms']                      # list of N strings
        self.augment          = augment
        self.noise_scale      = noise_scale

        N, C, D = self.image_embeddings.shape

        # Load augmented text variants if present (training augmentation)
        # Shape: (N, 6, 512) — 6 prompt variants per idiom
        self.text_embeddings_aug = data.get('text_embeddings_augmented', None)
        if self.text_embeddings_aug is not None:
            self.text_embeddings_aug = self.text_embeddings_aug.float()

        num_variants = self.text_embeddings_aug.shape[1] \
                       if self.text_embeddings_aug is not None else 1

        print(f"  Samples:           {N}")
        print(f"  Candidates/sample: {C}")
        print(f"  Embedding dim:     {D}")
        print(f"  Text variants:     {num_variants} per idiom "
              f"({'random pick each step' if augment else 'averaged'})")
        print(f"  Augmentation:      {'ON (training mode)' if augment else 'OFF (eval mode)'}")

        # Sanity check: make sure labels are valid
        assert (self.labels >= 0).all(), "Labels must be non-negative"
        assert (self.labels < C).all(),  f"Label index out of range (must be < {C})"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_embed   = self.text_embeddings[idx].clone()    # (512,)
        image_embeds = self.image_embeddings[idx].clone()   # (num_cands, 512)
        label        = self.labels[idx]                     # scalar

        if self.augment:
            # ── Augmentation 1: Random prompt variant selection ───────────────
            # Instead of always using the same averaged text embedding,
            # randomly pick one of the 6 pre-cached prompt variants.
            # Each training step sees a slightly different phrasing of the idiom.
            # Combined with shuffled batches: effectively 60 × 6 = 360 unique examples.
            if self.text_embeddings_aug is not None:
                num_variants = self.text_embeddings_aug.shape[1]         # 6
                variant_idx  = torch.randint(0, num_variants, (1,)).item()
                text_embed   = self.text_embeddings_aug[idx][variant_idx].clone()  # (512,)

            # ── Augmentation 2: Embedding-space noise ─────────────────────────
            # Small Gaussian noise keeps the model from memorizing exact values.
            # Applied AFTER variant selection so both augmentations compound.
            text_embed = text_embed + self.noise_scale * torch.randn_like(text_embed)
            text_embed = F.normalize(text_embed, dim=-1)

            image_embeds = image_embeds + self.noise_scale * torch.randn_like(image_embeds)
            image_embeds = F.normalize(image_embeds, dim=-1)

        return {
            'text_embed':   text_embed,    # (512,)
            'image_embeds': image_embeds,  # (num_cands, 512)
            'label':        label,         # scalar int
            'idiom':        self.idioms[idx]
        }

    def get_class_distribution(self):
        """Print distribution of labels — useful to check if dataset is balanced."""
        num_candidates = self.image_embeddings.shape[1]
        print(f"\n  Label distribution (should be roughly uniform across 0-{num_candidates-1}):")
        for i in range(num_candidates):
            count = (self.labels == i).sum().item()
            bar = "█" * count
            print(f"    Label {i}: {count:3d} samples  {bar}")
