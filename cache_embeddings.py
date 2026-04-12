"""
cache_embeddings.py
===================
PURPOSE:
    Run CLIP over your ENTIRE dataset exactly ONCE and save all embeddings to disk.

WHY THIS IS CRITICAL:
    CLIP takes ~2-3 seconds per image on CPU.
    With 71 idioms × 5 images = 355 images → ~15 minutes to process once.

    If we ran CLIP every training epoch:  50 epochs × 15 min = 12.5 HOURS
    With cached embeddings:               50 epochs × 1-2 min = ~1 HOUR

    This single script is the biggest performance improvement in Phase 1.

WHAT IT DOES:
    1. Loads CLIP ViT-B/32 (frozen — we never train it)
    2. Encodes every idiom sentence into a 512-dim text vector
    3. Encodes every candidate image into a 512-dim image vector
    4. Saves everything as .pt files in the cache/ folder
    5. CLIP can then be unloaded — training never needs it again

HOW TO RUN:
    python cache_embeddings.py

OUTPUT FILES:
    cache/task_a_train.pt
    cache/task_a_val.pt
    cache/task_b_train.pt   (if task_b data exists)
    cache/task_b_val.pt     (if task_b data exists)

EXPECTED JSON FORMAT (train.json / val.json):
    [
        {
            "idiom":    "kick the bucket",
            "sentence": "He finally kicked the bucket after a long illness.",
            "images":   ["001_0.jpg", "001_1.jpg", "001_2.jpg", "001_3.jpg", "001_4.jpg"],
            "label":    2    <-- index of the CORRECT image (0-based)
        },
        ...
    ]

    If your format is different, edit the load_dataset() function below.
"""

import os
import csv
import ast
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import config_phase1 as cfg


# ==============================================================================
# DATA LOADING — Reads your actual TSV format
# ==============================================================================

def load_task_a(tsv_path, images_root):
    """
    Load Task A annotations from a TSV file.

    TSV columns:
        compound, subset, sentence_type, sentence, expected_order,
        image1_name, image1_caption, image2_name, ..., image5_name, image5_caption

    How the label is extracted:
        expected_order is a ranked list like ['img3.png', 'img1.png', ...]
        The FIRST item is the most figuratively correct image.
        We find which of image1-5 matches that filename → that index is the label.

    Returns: list of dicts with keys:
        idiom, sentence, images (list of full paths), label (int 0-4)
    """
    data = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            idiom    = row['compound'].strip()
            sentence = row.get('sentence', idiom).strip() or idiom

            # Parse the ranked order list (stored as a Python list string)
            # e.g. "['img3.png', 'img1.png', ...]" → ['img3.png', 'img1.png', ...]
            try:
                expected_order = ast.literal_eval(row['expected_order'])
                correct_image  = expected_order[0]   # First = most correct
            except Exception:
                continue  # Skip malformed rows

            # Collect all 5 candidate image filenames in their original order
            images = []
            for i in range(1, 6):
                key = f'image{i}_name'
                if key in row and row[key].strip():
                    images.append(row[key].strip())

            if not images:
                continue

            # Find which index (0-4) corresponds to the correct image
            try:
                label = images.index(correct_image)
            except ValueError:
                continue  # Correct image not found in candidates — skip

            # Build full image paths: images_root/{idiom_folder}/{filename}
            # NOTE: TSV uses apostrophes (devil's advocate) but folders use
            # underscores (devil_s advocate) — replace ' with _
            idiom_folder = idiom.replace("'", "_")
            full_paths = [os.path.join(images_root, idiom_folder, img) for img in images]

            data.append({
                'idiom':    idiom,
                'sentence': sentence,
                'images':   full_paths,
                'label':    label
            })

    return data


def load_task_b(tsv_path, images_root):
    """
    Load Task B annotations from a TSV file.

    TSV columns:
        compound, subset, sentence_type, expected_item,
        sequence_caption1, sequence_caption2,
        image1_name, image1_caption, image2_name, ...

    How the label is extracted:
        expected_item is the filename of the correct image directly.
        We find its index among image1-4.

    Returns: list of dicts with keys:
        idiom, sentence, images (list of full paths), label (int 0-3)
    """
    data = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            idiom         = row['compound'].strip()
            correct_image = row['expected_item'].strip()

            # Use sequence_caption1 as the context sentence (describes the idiom scene)
            sentence = row.get('sequence_caption1', idiom).strip() or idiom

            # Collect all candidate image filenames
            images = []
            for i in range(1, 8):  # Check up to 7 image columns
                key = f'image{i}_name'
                if key in row and row[key].strip():
                    images.append(row[key].strip())

            if not images:
                continue

            try:
                label = images.index(correct_image)
            except ValueError:
                continue

            idiom_folder = idiom.replace("'", "_")
            full_paths = [os.path.join(images_root, idiom_folder, img) for img in images]

            data.append({
                'idiom':    idiom,
                'sentence': sentence,
                'images':   full_paths,
                'label':    label
            })

    return data


# ==============================================================================
# TEXT AUGMENTATION — Template-based sentence generation
# Multiplies your ~60 training samples by 6x at zero extra cost
# ==============================================================================

# Templates for generating varied sentence contexts per idiom.
# Each template produces a different surface form of the same figurative meaning.
# Having 6 variants means the model sees each idiom from 6 angles during training.
SENTENCE_TEMPLATES = [
    "An image that represents the expression '{idiom}'.",
    "A picture showing the figurative meaning of '{idiom}'.",
    "A visual representation of the idiom '{idiom}'.",
    "This image illustrates what it means to {idiom}.",
    "The phrase '{idiom}' is depicted in this image.",
    "{sentence}",   # Original context sentence (always included)
]

def make_augmented_prompts(idiom, sentence):
    """
    Generate 6 text prompt variants for an idiom using templates.

    WHY THIS HELPS:
        With only 60 training samples, the text encoder sees the same idiom
        phrased the same way every epoch. The model overfits to specific
        wording rather than learning the underlying figurative concept.

        By caching 6 variants and randomly selecting one per training step
        (done in dataset_phase1.py), we effectively have 60 × 6 = 360 training
        examples — a 6x multiplication for free.

    Returns:
        List of 6 prompt strings
    """
    prompts = []
    for template in SENTENCE_TEMPLATES:
        try:
            prompt = template.format(idiom=idiom, sentence=sentence)
        except KeyError:
            prompt = template.format(idiom=idiom)
        prompts.append(prompt)
    return prompts


def make_prompts(idiom, sentence):
    """
    Single averaged prompt (used for validation — deterministic).
    Averages all 6 variants into one stable embedding for consistent evaluation.
    """
    return make_augmented_prompts(idiom, sentence)


# ==============================================================================
# CORE CACHING FUNCTION
# ==============================================================================

def cache_split(data, split_name, model, preprocess, device):
    """
    Process one data split and return cached embeddings.

    Args:
        data:        list of dicts (from load_dataset)
        split_name:  string for progress display
        model:       frozen CLIP model
        preprocess:  CLIP image preprocessor
        device:      'cpu'

    Returns:
        dict with keys:
            'image_embeddings':       tensor (N, num_candidates, 512)
            'text_embeddings':        tensor (N, 512)      ← averaged, used for val
            'text_embeddings_augmented': tensor (N, 6, 512) ← all variants, used for train
            'labels':                 tensor (N,)
            'idioms':                 list of N strings

    TEXT AUGMENTATION STRATEGY:
        For training: we save all 6 prompt variants separately → (N, 6, 512)
            During training, dataset_phase1.py picks one variant randomly each step.
            Effect: 60 samples × 6 variants = 360 effective training examples.

        For validation: we save the average of all 6 → (N, 512)
            Averaging gives a more stable embedding for consistent evaluation.
            We always evaluate the same way so results are reproducible.
    """
    image_embeddings_all      = []
    text_embeddings_all       = []   # averaged (for val)
    text_embeddings_aug_all   = []   # all 6 variants (for train augmentation)
    labels_all                = []
    idioms_all                = []
    skipped                   = 0

    print(f"\n  Processing {split_name} ({len(data)} samples)...")

    for item in tqdm(data, desc=f"  {split_name}"):
        idiom    = item['idiom']
        sentence = item.get('sentence', idiom)
        label    = int(item['label'])
        images   = item['images']

        # ── Text Embeddings (6 augmented variants) ────────────────────────────
        prompts = make_augmented_prompts(idiom, sentence)               # list of 6 strings

        text_tokens = clip.tokenize(prompts, truncate=True).to(device)  # (6, 77)

        with torch.no_grad():
            text_feats = model.encode_text(text_tokens)                 # (6, 512)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # Store all 6 variants for training augmentation
        text_aug_embed = text_feats                                      # (6, 512)

        # Store averaged embedding for validation (stable, deterministic)
        text_embed = text_feats.mean(dim=0)                             # (512,)
        text_embed = text_embed / text_embed.norm()                     # re-normalize

        # ── Image Embeddings ──────────────────────────────────────────────────
        img_embeds = []

        for img_path in images:
            # Paths are already full absolute paths (built in load_task_a/b)
            full_path = img_path

            try:
                image = Image.open(full_path).convert("RGB")
                img_tensor = preprocess(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

                with torch.no_grad():
                    img_feat = model.encode_image(img_tensor)            # (1, 512)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                img_embeds.append(img_feat.squeeze(0))                   # (512,)

            except Exception as e:
                print(f"\n    ⚠ Could not load: {full_path} — {e}")
                # Replace missing image with zero vector (will score lowest)
                img_embeds.append(torch.zeros(cfg.EMBED_DIM, device=device))
                skipped += 1

        # Stack all candidate image embeddings: (num_candidates, 512)
        image_embed = torch.stack(img_embeds, dim=0)

        image_embeddings_all.append(image_embed)
        text_embeddings_all.append(text_embed)
        text_embeddings_aug_all.append(text_aug_embed)
        labels_all.append(label)
        idioms_all.append(idiom)

    # ── Combine into tensors ──────────────────────────────────────────────────
    image_embeddings      = torch.stack(image_embeddings_all,    dim=0).cpu()  # (N, C, 512)
    text_embeddings       = torch.stack(text_embeddings_all,     dim=0).cpu()  # (N, 512)
    text_embeddings_aug   = torch.stack(text_embeddings_aug_all, dim=0).cpu()  # (N, 6, 512)
    labels                = torch.tensor(labels_all, dtype=torch.long)          # (N,)

    if skipped > 0:
        print(f"\n  ⚠ {skipped} images could not be loaded and were replaced with zero vectors.")
        print(f"    Check image paths in your JSON if accuracy seems low.")

    print(f"\n  Text augmentation: {len(data)} idioms × 6 prompt variants "
          f"= {len(data) * 6} effective training texts")

    return {
        'image_embeddings':          image_embeddings,
        'text_embeddings':           text_embeddings,       # (N, 512)   — for val
        'text_embeddings_augmented': text_embeddings_aug,   # (N, 6, 512) — for train
        'labels':                    labels,
        'idioms':                    idioms_all
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("  CLIP Embedding Cache Builder")
    print("=" * 60)
    print(f"\n  Cache directory: {os.path.abspath(cfg.CACHE_DIR)}")

    # Load CLIP — only time it's needed
    print(f"\nLoading CLIP {cfg.CLIP_MODEL} on CPU...")
    print("(First run downloads ~350 MB — wait for it)\n")

    device = "cpu"
    model, preprocess = clip.load(cfg.CLIP_MODEL, device=device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print(f"✓ CLIP loaded and frozen.\n")

    # ── Task A ────────────────────────────────────────────────────────────────
    task_a_train_out = os.path.join(cfg.CACHE_DIR, "task_a_train.pt")
    task_a_val_out   = os.path.join(cfg.CACHE_DIR, "task_a_val.pt")

    if os.path.exists(cfg.TASK_A_TRAIN):
        if os.path.exists(task_a_train_out):
            print(f"Task A cache already exists — skipping. Delete cache/task_a_*.pt to rebuild.")
        else:
            print("Processing Task A...")
            # Use separate loaders — images live in different folders for train vs val
            train_data = load_task_a(cfg.TASK_A_TRAIN, cfg.TASK_A_TRAIN_IMG)
            val_data   = load_task_a(cfg.TASK_A_VAL,   cfg.TASK_A_VAL_IMG)

            print(f"  Loaded {len(train_data)} train samples, {len(val_data)} val samples")

            train_cache = cache_split(train_data, "Task A Train", model, preprocess, device)
            val_cache   = cache_split(val_data,   "Task A Val",   model, preprocess, device)

            torch.save(train_cache, task_a_train_out)
            torch.save(val_cache,   task_a_val_out)

            print(f"\n  ✓ Task A cached:")
            print(f"    {task_a_train_out}")
            print(f"    {task_a_val_out}")
    else:
        print(f"  Task A TSV not found at: {cfg.TASK_A_TRAIN}")

    # ── Task B ────────────────────────────────────────────────────────────────
    task_b_train_out = os.path.join(cfg.CACHE_DIR, "task_b_train.pt")
    task_b_val_out   = os.path.join(cfg.CACHE_DIR, "task_b_val.pt")

    if os.path.exists(cfg.TASK_B_TRAIN):
        if os.path.exists(task_b_train_out):
            print(f"\nTask B cache already exists — skipping. Delete cache/task_b_*.pt to rebuild.")
        else:
            print("\nProcessing Task B...")
            train_data = load_task_b(cfg.TASK_B_TRAIN, cfg.TASK_B_TRAIN_IMG)
            val_data   = load_task_b(cfg.TASK_B_VAL,   cfg.TASK_B_VAL_IMG)

            print(f"  Loaded {len(train_data)} train samples, {len(val_data)} val samples")

            train_cache = cache_split(train_data, "Task B Train", model, preprocess, device)
            val_cache   = cache_split(val_data,   "Task B Val",   model, preprocess, device)

            torch.save(train_cache, task_b_train_out)
            torch.save(val_cache,   task_b_val_out)

            print(f"\n  ✓ Task B cached:")
            print(f"    {task_b_train_out}")
            print(f"    {task_b_val_out}")
    else:
        print(f"\n  Task B TSV not found at: {cfg.TASK_B_TRAIN}")

    print("\n" + "=" * 60)
    print("  Caching complete!")
    print("  Next step: python train_phase1.py --task a")
    print("=" * 60)


if __name__ == "__main__":
    main()
