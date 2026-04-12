"""
train_phase2.py
===============
PURPOSE:
    Train the Phase 2 model on pre-cached Phase 2 embeddings.

WHAT'S NEW VS PHASE 1:
    - Loads phase2 cache (iapd_embeddings + conceptnet_vectors + image_embeddings)
    - Uses Phase2FigurativeModel (IAPD attention + ConceptNet fusion + adapters)
    - Logs IAPD attention weights so you can see which perspective the model uses
    - Same combined loss (cross-entropy + margin ranking)
    - Same early stopping strategy

HOW TO RUN:
    python train_phase2.py --task a
    python train_phase2.py --task b
    python train_phase2.py --task both

EXPECTED RESULTS:
    Task A: 55–65% (up from Phase 1's 53.3%)
    Task B: 58–68%
    Training time: ~3–5 min/epoch on CPU
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import config_phase1 as cfg
from phase2_model import Phase2FigurativeModel


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==============================================================================
# DATASET
# ==============================================================================

class Phase2Dataset(Dataset):
    """
    Loads Phase 2 cached embeddings.

    Each sample contains:
        iapd_embeddings:    (3, 512) — literal / figurative / contextual
        image_embeddings:   (num_candidates, 512)
        conceptnet_vectors: (300,)
        label:              int
    """

    def __init__(self, cache_path, augment=False, noise_scale=0.01):
        print(f"\nLoading Phase 2 dataset: {cache_path}")
        data = torch.load(cache_path, map_location="cpu")

        self.iapd_embeddings   = data["iapd_embeddings"].float()    # (N, 3, 512)
        self.image_embeddings  = data["image_embeddings"].float()   # (N, C, 512)
        self.conceptnet_vectors = data["conceptnet_vectors"].float() # (N, 300)
        self.labels            = data["labels"]                      # (N,)
        self.idioms            = data["idioms"]
        self.augment           = augment
        self.noise_scale       = noise_scale

        N, P, D = self.iapd_embeddings.shape
        print(f"  Samples:           {N}")
        print(f"  Perspectives:      {P}  (literal / figurative / contextual)")
        print(f"  Image candidates:  {self.image_embeddings.shape[1]}")
        print(f"  ConceptNet dim:    {self.conceptnet_vectors.shape[1]}")
        print(f"  Augmentation:      {'ON' if augment else 'OFF'}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        iapd   = self.iapd_embeddings[idx].clone()    # (3, 512)
        images = self.image_embeddings[idx].clone()   # (C, 512)
        cn_vec = self.conceptnet_vectors[idx].clone() # (300,)
        label  = self.labels[idx]

        if self.augment:
            # Small noise on IAPD embeddings and images
            iapd   = iapd   + self.noise_scale * torch.randn_like(iapd)
            iapd   = torch.nn.functional.normalize(iapd, dim=-1)
            images = images + self.noise_scale * torch.randn_like(images)
            images = torch.nn.functional.normalize(images, dim=-1)
            # ConceptNet vectors are not augmented — they're fixed commonsense knowledge

        return {
            "iapd_embeddings":   iapd,
            "image_embeddings":  images,
            "conceptnet_vectors": cn_vec,
            "label":             label,
            "idiom":             self.idioms[idx]
        }


# ==============================================================================
# LOSS FUNCTION (same combined CE + margin as Phase 1)
# ==============================================================================

class RankingLoss(nn.Module):
    def __init__(self, margin=0.2, margin_weight=0.3):
        super().__init__()
        self.ce_loss       = nn.CrossEntropyLoss()
        self.margin        = margin
        self.margin_weight = margin_weight

    def forward(self, scores, labels):
        ce = self.ce_loss(scores, labels)
        B  = scores.shape[0]
        N  = scores.shape[1]

        correct_scores = scores[torch.arange(B), labels]
        margin_losses  = []
        for i in range(N):
            wrong_scores = scores[:, i]
            pair_loss    = torch.relu(wrong_scores + self.margin - correct_scores)
            is_wrong     = (i != labels.cpu()).float().to(scores.device)
            margin_losses.append((pair_loss * is_wrong).mean())

        margin = torch.stack(margin_losses).mean()
        return ce + self.margin_weight * margin, ce, margin


# ==============================================================================
# EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total   = 0
    all_preds  = []
    all_labels = []
    all_attn   = []  # IAPD attention weights — shows which perspective model used

    for batch in dataloader:
        iapd   = batch["iapd_embeddings"].to(device)
        images = batch["image_embeddings"].to(device)
        cn_vec = batch["conceptnet_vectors"].to(device)
        labels = batch["label"].to(device)

        scores, attn_weights = model(iapd, images, cn_vec)
        preds = scores.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_attn.append(attn_weights.cpu())

    accuracy   = correct / total if total > 0 else 0.0
    avg_attn   = torch.cat(all_attn, dim=0).mean(dim=0)  # (3,) average attention

    return accuracy, all_preds, all_labels, avg_attn


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(task="a"):
    set_seed(cfg.SEED)
    device = torch.device("cpu")

    task_key     = task.lower()
    train_cache  = os.path.join(cfg.CACHE_DIR, f"phase2_task_{task_key}_train.pt")
    val_cache    = os.path.join(cfg.CACHE_DIR, f"phase2_task_{task_key}_val.pt")

    if not os.path.exists(train_cache):
        raise FileNotFoundError(
            f"Phase 2 cache not found: {train_cache}\n"
            f"Run: python cache_phase2.py"
        )

    print(f"\n{'='*60}")
    print(f"  Training Phase 2 — Task {task.upper()}")
    print(f"{'='*60}")

    # Datasets
    train_ds = Phase2Dataset(train_cache, augment=True)
    val_ds   = Phase2Dataset(val_cache,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=4,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4,
                              shuffle=False, num_workers=0)

    # Model — only 4 learnable parameters (3 IAPD weights + 1 CN gate)
    # This prevents the catastrophic overfitting seen with 813K-param Phase 2 v1
    model = Phase2FigurativeModel(
        embed_dim=512, cn_dim=300, temperature=0.07
    ).to(device)
    model.count_parameters()

    # Optimizer and scheduler
    # Higher LR (5e-3) because there are only 4 params — converges quickly
    # No weight decay — regularising 4 scalars is unnecessary and harmful
    criterion = RankingLoss(margin=0.2, margin_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=300, eta_min=1e-4
    )

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    best_path    = os.path.join(cfg.MODEL_DIR, f"best_phase2_task_{task_key}.pt")

    best_val_acc     = 0.0
    best_epoch       = 0
    patience_counter = 0
    PATIENCE         = 50   # more patience — 4-param model converges slowly
    EPOCHS           = 300  # more epochs — same reason

    print(f"\n  Epoch | Train Loss | Train Acc |  Val Acc |     Best |   Lit  |  Fig  |  Ctx")
    print(f"  {'-'*85}")
    print(f"  (Model has only 4 learnable params — no overfitting expected)")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0

        for batch in train_loader:
            iapd   = batch["iapd_embeddings"].to(device)
            images = batch["image_embeddings"].to(device)
            cn_vec = batch["conceptnet_vectors"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            scores, _ = model(iapd, images, cn_vec)
            loss, _, _ = criterion(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss    += loss.item() * labels.size(0)
            preds          = scores.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total   += labels.size(0)

        scheduler.step()

        avg_loss  = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_acc, _, _, avg_attn = evaluate(model, val_loader, device)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc     = val_acc
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_acc": val_acc
            }, best_path)
        else:
            patience_counter += 1

        # Log every 5 epochs, always log best
        if epoch % 5 == 0 or is_best or epoch == 1:
            lr  = scheduler.get_last_lr()[0]
            lit = avg_attn[0].item()
            fig = avg_attn[1].item()
            ctx = avg_attn[2].item()
            marker = " ← BEST" if is_best else ""
            print(f"  {epoch:>6} | {avg_loss:>10.4f} | {train_acc:>8.1%} | "
                  f"{val_acc:>7.1%} | {best_val_acc:>7.1%} | "
                  f"{lit:.2f} | {fig:.2f} | {ctx:.2f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Final report
    print(f"\n{'='*60}")
    print(f"  Phase 2 Training Complete — Task {task.upper()}")
    print(f"{'='*60}")
    print(f"\n  Best Val Accuracy:    {best_val_acc:.1%}  (epoch {best_epoch})")
    print(f"  Phase 1 Accuracy:     ~53.3%")
    print(f"  Baseline (IRFL):       ~22.0%")
    print(f"  Phase 2 improvement:  {(best_val_acc - 0.533)*100:+.1f} pp over Phase 1")
    print(f"  Total improvement:    {(best_val_acc - 0.22)*100:+.1f} pp over baseline")
    print(f"\n  Best model: {best_path}")

    # Load best and show IAPD attention analysis
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    val_acc, preds, labels, avg_attn = evaluate(model, val_loader, device)

    print(f"\n  IAPD Perspective Weights (what the model focuses on):")
    print(f"    Literal perspective:     {avg_attn[0]:.1%}")
    print(f"    Figurative perspective:  {avg_attn[1]:.1%}  ← should be highest")
    print(f"    Contextual perspective:  {avg_attn[2]:.1%}")
    print(f"\n  Overall Val Accuracy: {val_acc:.1%}")

    return best_val_acc


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="a",
                        choices=["a", "b", "both"])
    args = parser.parse_args()

    if args.task == "both":
        acc_a = train("a")
        acc_b = train("b")
        print(f"\n  Final: Task A = {acc_a:.1%}, Task B = {acc_b:.1%}")
    else:
        train(args.task)
