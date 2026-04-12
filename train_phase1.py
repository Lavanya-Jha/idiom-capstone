"""
train_phase1.py
===============
PURPOSE:
    Train the FigurativeAdapter on top of pre-cached CLIP embeddings.

WHAT THIS SCRIPT DOES:
    1. Loads cached CLIP embeddings from cache/*.pt
    2. Trains only the lightweight adapter (~265K params) — CLIP never touched
    3. Uses cross-entropy loss over image similarity scores (standard classification)
    4. Early stopping to prevent overfitting on tiny dataset
    5. Saves the best checkpoint based on validation accuracy
    6. Prints a training curve and final evaluation report

HOW TO RUN:
    # Train on Task A only:
    python train_phase1.py --task a

    # Train on Task B only:
    python train_phase1.py --task b

    # Train on both:
    python train_phase1.py --task both

EXPECTED RESULTS (Phase 1):
    Task A: 30–38% accuracy  (vs 22% baseline)
    Task B: 35–42% accuracy
    Training time: ~2–5 minutes per epoch on CPU

LOSS FUNCTION — Why Cross-Entropy?
    For each idiom we have N candidate images and must pick the correct one.
    The adapter outputs a score for each candidate.
    Cross-entropy treats this as N-way classification:
        loss = -log(softmax(scores)[correct_label])
    This is simple, stable on small datasets, and directly optimizes accuracy.

    In Phase 3 we'll add contrastive/curriculum losses on top of this.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config_phase1 as cfg
from adapter_model import FigurativeAdapter
from dataset_phase1 import CachedEmbeddingDataset


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==============================================================================
# LOSS FUNCTION
# ==============================================================================

class RankingLoss(nn.Module):
    """
    Combined loss: Cross-Entropy + Margin Ranking.

    Cross-Entropy (primary):
        Treats image selection as N-way classification.
        Directly optimizes: "give the correct image the highest score"

    Margin Ranking (secondary, weight=0.3):
        Ensures the correct image scores HIGHER than each wrong image
        by at least a margin of 0.2.
        This explicitly penalizes cases where a wrong image nearly ties
        with the correct one — important for the partial-literal confusion problem.

    Total loss = CE_loss + 0.3 * margin_loss
    """

    def __init__(self, margin=0.2, margin_weight=0.3):
        super().__init__()
        self.ce_loss      = nn.CrossEntropyLoss()
        self.margin       = margin
        self.margin_weight = margin_weight

    def forward(self, scores, labels):
        """
        Args:
            scores: (batch, num_candidates) — raw similarity scores from adapter
            labels: (batch,) — index of correct image

        Returns:
            total_loss: scalar
            ce_loss:    scalar (for logging)
            margin_loss: scalar (for logging)
        """
        # ── Cross-Entropy Loss ────────────────────────────────────────────────
        ce = self.ce_loss(scores, labels)

        # ── Margin Ranking Loss ───────────────────────────────────────────────
        # For each sample, get the score of the CORRECT image
        batch_size    = scores.shape[0]
        num_candidates = scores.shape[1]

        # Gather correct image scores: (batch,)
        correct_scores = scores[torch.arange(batch_size), labels]

        # Compare correct score against each wrong image score
        margin_losses = []
        for i in range(num_candidates):
            # Skip comparison with itself
            wrong_mask = (torch.arange(num_candidates) != labels.unsqueeze(1)).float()

            wrong_scores = scores[:, i]

            # We want: correct_score > wrong_score + margin
            # Loss = max(0, wrong_score + margin - correct_score)
            pair_loss = torch.relu(wrong_scores + self.margin - correct_scores)

            # Only count this if image i is actually a wrong image for that sample
            is_wrong = (i != labels.cpu()).float().to(scores.device)
            margin_losses.append((pair_loss * is_wrong).mean())

        margin = torch.stack(margin_losses).mean()

        total = ce + self.margin_weight * margin

        return total, ce, margin


# ==============================================================================
# EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Compute accuracy on a dataset split.

    Accuracy = fraction of idioms where the highest-scoring image is correct.
    This matches the evaluation metric used in the IRFL paper.
    """
    model.eval()

    correct = 0
    total   = 0
    all_predictions = []
    all_labels      = []

    for batch in dataloader:
        text_embeds  = batch['text_embed'].to(device)    # (B, 512)
        image_embeds = batch['image_embeds'].to(device)  # (B, N, 512)
        labels       = batch['label'].to(device)         # (B,)

        scores      = model(text_embeds, image_embeds)   # (B, N)
        predictions = scores.argmax(dim=-1)              # (B,)

        correct += (predictions == labels).sum().item()
        total   += labels.size(0)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = correct / total if total > 0 else 0.0

    return accuracy, all_predictions, all_labels


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(task='a'):
    set_seed(cfg.SEED)

    device = torch.device('cpu')

    # ── Determine cache paths ─────────────────────────────────────────────────
    task_key = task.lower()
    train_cache = os.path.join(cfg.CACHE_DIR, f"task_{task_key}_train.pt")
    val_cache   = os.path.join(cfg.CACHE_DIR, f"task_{task_key}_val.pt")

    if not os.path.exists(train_cache):
        raise FileNotFoundError(
            f"Cache file not found: {train_cache}\n"
            f"Run cache_embeddings.py first!"
        )

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Phase 1 — Task {task.upper()}")
    print(f"{'='*60}")

    train_dataset = CachedEmbeddingDataset(train_cache, augment=True)
    val_dataset   = CachedEmbeddingDataset(val_cache,   augment=False)

    train_dataset.get_class_distribution()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,                   # Shuffle training data each epoch
        num_workers=cfg.NUM_WORKERS,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,                  # Never shuffle validation
        num_workers=cfg.NUM_WORKERS
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FigurativeAdapter(
        embed_dim=cfg.EMBED_DIM,
        bottleneck_dim=cfg.ADAPTER_DIM,
        temperature=cfg.TEMPERATURE
    ).to(device)

    model.count_parameters()

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = RankingLoss(margin=0.2, margin_weight=0.3)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # Cosine annealing: smoothly reduces LR from cfg.LR to cfg.LR/10 over all epochs
    # This helps the model fine-tune in later epochs without overshooting
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS,
        eta_min=cfg.LR / 10
    )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(cfg.MODEL_DIR, f"best_adapter_task_{task_key}.pt")

    # ── Training State ────────────────────────────────────────────────────────
    best_val_acc     = 0.0
    best_epoch       = 0
    patience_counter = 0
    history          = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n  Training for up to {cfg.EPOCHS} epochs (early stop after "
          f"{cfg.EARLY_STOP_PATIENCE} epochs without improvement)")
    print(f"  Batch size: {cfg.BATCH_SIZE} | LR: {cfg.LR} | Device: CPU\n")
    print(f"  {'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Acc':>8} | {'Best':>8} | {'LR':>8}")
    print(f"  {'-'*65}")

    # ── Main Training Loop ────────────────────────────────────────────────────
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()

        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0

        for batch in train_loader:
            text_embeds  = batch['text_embed'].to(device)
            image_embeds = batch['image_embeds'].to(device)
            labels       = batch['label'].to(device)

            optimizer.zero_grad()

            scores = model(text_embeds, image_embeds)               # (B, N)
            loss, ce_loss, margin_loss = criterion(scores, labels)

            loss.backward()

            # Gradient clipping: prevents exploding gradients (rare but possible)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Track metrics
            epoch_loss    += loss.item() * labels.size(0)
            predictions    = scores.argmax(dim=-1)
            epoch_correct += (predictions == labels).sum().item()
            epoch_total   += labels.size(0)

        scheduler.step()

        # ── Compute epoch metrics ─────────────────────────────────────────────
        avg_loss  = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_acc, _, _ = evaluate(model, val_loader, device)

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # ── Early stopping & checkpoint ───────────────────────────────────────
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc     = val_acc
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':    val_acc,
                'config':     {
                    'embed_dim':      cfg.EMBED_DIM,
                    'bottleneck_dim': cfg.ADAPTER_DIM,
                    'temperature':    cfg.TEMPERATURE
                }
            }, best_model_path)
        else:
            patience_counter += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if epoch % cfg.LOG_EVERY == 0 or is_best or epoch == 1:
            current_lr = scheduler.get_last_lr()[0]
            best_marker = " ← BEST" if is_best else ""
            print(f"  {epoch:>6} | {avg_loss:>10.4f} | {train_acc:>8.1%} | "
                  f"{val_acc:>7.1%} | {best_val_acc:>7.1%} |"
                  f" {current_lr:.2e}{best_marker}")

        # ── Early stop check ──────────────────────────────────────────────────
        if patience_counter >= cfg.EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {cfg.EARLY_STOP_PATIENCE} epochs)")
            break

    # ── Final Report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Complete — Task {task.upper()}")
    print(f"{'='*60}")
    print(f"\n  Best Validation Accuracy: {best_val_acc:.1%}  (epoch {best_epoch})")
    print(f"  Baseline (IRFL CLIP):       ~22.0%")
    print(f"  Improvement:               +{(best_val_acc - 0.22)*100:.1f} percentage points")
    print(f"\n  Best model saved to: {best_model_path}")

    # ── Load best model and run final detailed evaluation ─────────────────────
    print(f"\n  Running detailed evaluation on best model...")

    checkpoint = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    val_acc, predictions, labels = evaluate(model, val_loader, device)

    # Per-class accuracy breakdown
    num_candidates = val_dataset[0]['image_embeds'].shape[0]
    print(f"\n  Accuracy by correct label position (shows if model has label bias):")
    for pos in range(num_candidates):
        pos_mask   = [l == pos for l in labels]
        pos_preds  = [p for p, m in zip(predictions, pos_mask) if m]
        pos_labels = [l for l, m in zip(labels,      pos_mask) if m]
        if pos_labels:
            pos_acc = sum(p == l for p, l in zip(pos_preds, pos_labels)) / len(pos_labels)
            print(f"    Position {pos}: {pos_acc:.1%}  ({len(pos_labels)} samples)")

    print(f"\n  Overall Val Accuracy: {val_acc:.1%}")

    return best_val_acc, history


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 1 Figurative Adapter")

    parser.add_argument(
        "--task",
        type=str,
        default="a",
        choices=["a", "b", "both"],
        help="Which task to train on: 'a', 'b', or 'both' (default: 'a')"
    )

    args = parser.parse_args()

    if args.task == "both":
        print("\n>>> Training Task A <<<")
        acc_a, hist_a = train(task='a')

        print("\n>>> Training Task B <<<")
        acc_b, hist_b = train(task='b')

        print(f"\n{'='*60}")
        print(f"  Final Summary")
        print(f"{'='*60}")
        print(f"  Task A Best Val Accuracy: {acc_a:.1%}")
        print(f"  Task B Best Val Accuracy: {acc_b:.1%}")

    else:
        train(task=args.task)
