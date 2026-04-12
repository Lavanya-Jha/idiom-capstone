"""
cache_phase3.py
===============
PURPOSE:
    Build Phase 3 cache by adding TWO new signals on top of Phase 2:

    1. CAPTION EMBEDDINGS
       The augmented_data TSV files contain a text caption for every image
       (e.g. "The image shows a person scrubbing a stove with a sponge").
       Encoding these captions with CLIP gives a TEXT representation of each
       image — which CLIP can compare to idiom text far more reliably than
       comparing text to raw image pixels.

       Why better? CLIP text-to-text similarity avoids the visual-semantic gap.
       The model can reason: "caption mentions scrubbing → matches figurative
       meaning of 'elbow grease' (hard work)" — something pixel matching misses.

    2. FIGURATIVE–LITERAL GAP EMBEDDINGS
       For each image we pre-compute:
           gap = sim(figurative_text, image) − sim(literal_text, image)
       The correct figurative image should have a HIGH gap (figurative >> literal).
       Literal distractor images (showing the actual words) have a LOW/NEGATIVE gap.
       This is stored as a (num_candidates,) vector per sample.

HOW TO RUN:
    python cache_phase3.py

OUTPUT:
    cache/phase3_task_a_train.pt
    cache/phase3_task_a_val.pt
    cache/phase3_task_b_train.pt  (if available)
    cache/phase3_task_b_val.pt    (if available)

EACH CACHE CONTAINS:
    iapd_embeddings:    (N, 3, 512)  — same as Phase 2
    image_embeddings:   (N, C, 512)  — same as Phase 2
    caption_embeddings: (N, C, 512)  — NEW: CLIP-encoded image captions
    fig_lit_gap:        (N, C)       — NEW: figurative−literal similarity gap
    conceptnet_vectors: (N, 300)     — same as Phase 2
    labels:             (N,)
    idioms:             list[str]
"""

import os
import csv
import ast
import pickle
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
import config_phase1 as cfg


# ==============================================================================
# PATHS
# ==============================================================================

AUGMENTED_A_TRAIN = os.path.join(cfg.DATA_ROOT, "augmented_data", "subtask_a_train_augmented.tsv")
AUGMENTED_A_VAL   = os.path.join(cfg.DATA_ROOT, "augmented_data", "subtask_a_dev_augmented.tsv")
AUGMENTED_B_TRAIN = os.path.join(cfg.DATA_ROOT, "augmented_data", "subtask_b_train_augmented.tsv")
AUGMENTED_B_VAL   = os.path.join(cfg.DATA_ROOT, "augmented_data", "subtask_b_dev_augmented.tsv")

CONCEPTNET_PKL    = os.path.join(os.path.dirname(__file__), "conceptnet", "numberbatch_en.pkl")
CACHE_DIR         = os.path.join(os.path.dirname(__file__), "cache")


# ==============================================================================
# IAPD PROMPT GENERATION (same as Phase 2)
# ==============================================================================

def generate_iapd_prompts(idiom, sentence):
    words = idiom.lower().split()
    if len(words) >= 2:
        literal = f"a photo showing {' and '.join(words[:3])}"
    else:
        literal = f"a photo of {idiom}"
    figurative = f"an image representing the figurative meaning of the expression '{idiom}'"
    contextual  = sentence if sentence and sentence != idiom \
                  else f"a visual scene depicting '{idiom}'"
    return {"literal": literal, "figurative": figurative, "contextual": contextual}


# ==============================================================================
# DATA LOADING — reads augmented TSV with captions
# ==============================================================================

def load_augmented_task_a(tsv_path, images_root):
    """
    Load Task A augmented TSV.
    Returns unique idioms only (augmented TSV repeats each idiom multiple times).
    Each entry includes per-image captions.
    """
    seen   = set()
    data   = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idiom = row["compound"].strip()
            if idiom in seen:
                continue          # keep only first occurrence per idiom
            seen.add(idiom)

            sentence = row.get("sentence", idiom).strip() or idiom
            try:
                expected_order = ast.literal_eval(row["expected_order"])
                correct_image  = expected_order[0]
            except Exception:
                continue

            images   = []
            captions = []
            for i in range(1, 6):
                img_key = f"image{i}_name"
                cap_key = f"image{i}_caption"
                if img_key in row and row[img_key].strip():
                    images.append(row[img_key].strip())
                    captions.append(row.get(cap_key, "").strip())

            if not images:
                continue
            try:
                label = images.index(correct_image)
            except ValueError:
                continue

            # Try multiple folder naming conventions
            for idiom_folder in [idiom, idiom.replace("'", "_"), idiom.replace("'", "")]:
                if os.path.exists(os.path.join(images_root, idiom_folder, images[0])):
                    break

            full_paths = [os.path.join(images_root, idiom_folder, img) for img in images]

            data.append({
                "idiom":    idiom,
                "sentence": sentence,
                "images":   full_paths,
                "captions": captions,
                "label":    label
            })
    return data


def load_augmented_task_b(tsv_path, images_root):
    """Load Task B augmented TSV with captions."""
    seen = set()
    data = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idiom = row["compound"].strip()
            if idiom in seen:
                continue
            seen.add(idiom)

            correct_image = row.get("expected_item", "").strip()
            sentence      = row.get("sequence_caption1", idiom).strip() or idiom

            images   = []
            captions = []
            for i in range(1, 8):
                img_key = f"image{i}_name"
                cap_key = f"image{i}_caption"
                if img_key in row and row[img_key].strip():
                    images.append(row[img_key].strip())
                    captions.append(row.get(cap_key, "").strip())

            if not images:
                continue
            try:
                label = images.index(correct_image)
            except ValueError:
                continue

            for idiom_folder in [idiom, idiom.replace("'", "_"), idiom.replace("'", "")]:
                if os.path.exists(os.path.join(images_root, idiom_folder, images[0])):
                    break

            full_paths = [os.path.join(images_root, idiom_folder, img) for img in images]
            data.append({
                "idiom":    idiom,
                "sentence": sentence,
                "images":   full_paths,
                "captions": captions,
                "label":    label
            })
    return data


# ==============================================================================
# CONCEPTNET LOADER
# ==============================================================================

def load_conceptnet(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"  ⚠ ConceptNet not found — skipping CN vectors")
        return None
    with open(pkl_path, "rb") as f:
        embeddings = pickle.load(f)
    print(f"  ✓ ConceptNet loaded ({len(embeddings):,} words)")
    return embeddings


def get_conceptnet_vector(idiom, cn_embeddings):
    stopwords = {"a", "an", "the", "of", "in", "on", "at", "to", "for",
                 "with", "and", "or", "but", "is", "are", "was", "were"}
    words = [w.lower().strip("'s") for w in idiom.split()
             if w.lower() not in stopwords]
    vecs = []
    for word in words:
        if word in cn_embeddings:
            vecs.append(cn_embeddings[word])
        elif idiom.replace(" ", "_").lower() in cn_embeddings:
            vecs.append(cn_embeddings[idiom.replace(" ", "_").lower()])
    if vecs:
        avg = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg
    return np.zeros(300, dtype=np.float32)


# ==============================================================================
# CORE CACHING — Phase 3 adds caption embeddings and fig-lit gap
# ==============================================================================

def cache_split_phase3(data, split_name, clip_model, preprocess, cn_embeddings, device):
    """
    Cache Phase 3 embeddings for one split.

    New vs Phase 2:
        caption_embeddings: (N, C, 512) — CLIP-encoded image captions
        fig_lit_gap:        (N, C)      — figurative − literal similarity per image
    """
    iapd_all     = []
    image_all    = []
    caption_all  = []
    gap_all      = []
    cn_all       = []
    labels_all   = []
    idioms_all   = []
    skipped      = 0

    print(f"\n  Processing {split_name} ({len(data)} samples)...")

    for item in tqdm(data, desc=f"  {split_name}"):
        idiom    = item["idiom"]
        sentence = item.get("sentence", idiom)
        label    = int(item["label"])
        images   = item["images"]
        captions = item["captions"]

        # ── IAPD text embeddings (3 perspectives) ────────────────────────────
        perspectives = generate_iapd_prompts(idiom, sentence)
        prompts = [perspectives["literal"], perspectives["figurative"], perspectives["contextual"]]
        text_tokens = clip.tokenize(prompts, truncate=True).to(device)
        with torch.no_grad():
            text_feats = clip_model.encode_text(text_tokens)           # (3, 512)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        iapd_embed = text_feats   # (3, 512)

        lit_embed = text_feats[0]  # (512,) — literal perspective
        fig_embed = text_feats[1]  # (512,) — figurative perspective

        # ── ConceptNet vector ─────────────────────────────────────────────────
        if cn_embeddings:
            cn_vec = get_conceptnet_vector(idiom, cn_embeddings)
        else:
            cn_vec = np.zeros(300, dtype=np.float32)

        # ── Image embeddings + caption embeddings + fig-lit gap ───────────────
        img_embeds     = []
        cap_embeds     = []
        gap_scores     = []

        for img_path, caption in zip(images, captions):
            # Image embedding
            try:
                image      = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_feat = clip_model.encode_image(img_tensor)     # (1, 512)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                img_embed = img_feat.squeeze(0)                         # (512,)
            except Exception as e:
                print(f"\n    ⚠ Image load failed: {img_path} — {e}")
                img_embed = torch.zeros(cfg.EMBED_DIM, device=device)
                skipped += 1

            # Caption embedding — text description of image contents
            cap_text = caption if caption else f"an image related to {idiom}"
            cap_token = clip.tokenize([cap_text], truncate=True).to(device)
            with torch.no_grad():
                cap_feat = clip_model.encode_text(cap_token)           # (1, 512)
                cap_feat = cap_feat / cap_feat.norm(dim=-1, keepdim=True)
            cap_embed = cap_feat.squeeze(0)                             # (512,)

            # Figurative–Literal Gap: sim(fig, image) − sim(lit, image)
            # High gap = image aligns with figurative meaning, not literal words
            fig_sim = (fig_embed * img_embed).sum().item()
            lit_sim = (lit_embed * img_embed).sum().item()
            gap     = fig_sim - lit_sim

            img_embeds.append(img_embed)
            cap_embeds.append(cap_embed)
            gap_scores.append(gap)

        iapd_all.append(iapd_embed)
        image_all.append(torch.stack(img_embeds, dim=0))    # (C, 512)
        caption_all.append(torch.stack(cap_embeds, dim=0))  # (C, 512)
        gap_all.append(torch.tensor(gap_scores))             # (C,)
        cn_all.append(torch.tensor(cn_vec))
        labels_all.append(label)
        idioms_all.append(idiom)

    if skipped > 0:
        print(f"\n  ⚠ {skipped} images skipped")

    return {
        "iapd_embeddings":    torch.stack(iapd_all,    dim=0).cpu(),  # (N, 3, 512)
        "image_embeddings":   torch.stack(image_all,   dim=0).cpu(),  # (N, C, 512)
        "caption_embeddings": torch.stack(caption_all, dim=0).cpu(),  # (N, C, 512)
        "fig_lit_gap":        torch.stack(gap_all,     dim=0).cpu(),  # (N, C)
        "conceptnet_vectors": torch.stack(cn_all,      dim=0).cpu(),  # (N, 300)
        "labels":             torch.tensor(labels_all, dtype=torch.long),
        "idioms":             idioms_all
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("  Phase 3 Embedding Cache Builder")
    print("=" * 60)
    print("  New signals vs Phase 2:")
    print("    + Caption embeddings (text descriptions of images)")
    print("    + Figurative-Literal gap scores per image")

    # Load CLIP
    print(f"\nLoading CLIP {cfg.CLIP_MODEL} on CPU...")
    device = "cpu"
    clip_model, preprocess = clip.load(cfg.CLIP_MODEL, device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print("✓ CLIP loaded and frozen.")

    # Load ConceptNet
    cn_embeddings = load_conceptnet(CONCEPTNET_PKL)

    # ── Task A ────────────────────────────────────────────────────────────────
    p3_a_train = os.path.join(CACHE_DIR, "phase3_task_a_train.pt")
    p3_a_val   = os.path.join(CACHE_DIR, "phase3_task_a_val.pt")

    if os.path.exists(p3_a_train):
        print("\nPhase 3 Task A cache already exists. Delete to rebuild.")
    elif os.path.exists(AUGMENTED_A_TRAIN):
        print("\nProcessing Task A...")
        train_data = load_augmented_task_a(AUGMENTED_A_TRAIN, cfg.TASK_A_TRAIN_IMG)
        val_data   = load_augmented_task_a(AUGMENTED_A_VAL,   cfg.TASK_A_VAL_IMG)

        # ── Merge Extended idioms — Extended TSV has full captions too ──────────
        # NOTE: subtask_a_xe.tsv has the same columns as Train/Dev including
        #       image{i}_caption. The previous version incorrectly used empty
        #       strings; now we read the real captions.
        EXTENDED_TSV = os.path.join(cfg.DATA_ROOT, "Subtask A", "EN", "Extended", "xeval", "subtask_a_xe.tsv")
        EXTENDED_IMG = os.path.join(cfg.DATA_ROOT, "Subtask A", "EN", "Extended", "xeval")
        if os.path.exists(EXTENDED_TSV):
            import csv as _csv, ast as _ast
            # Only add idioms NOT already in train_data (avoid leakage from dev overlap)
            seen_idioms = {item["idiom"] for item in train_data}
            added = 0
            with open(EXTENDED_TSV, "r", encoding="utf-8") as f:
                reader = _csv.DictReader(f, delimiter="\t")
                for row in reader:
                    idiom = row["compound"].strip()
                    if idiom in seen_idioms:
                        continue
                    seen_idioms.add(idiom)
                    sentence = row.get("sentence", idiom).strip() or idiom
                    try:
                        expected_order = _ast.literal_eval(row["expected_order"])
                        correct_image  = expected_order[0]
                    except Exception:
                        continue
                    images = [row[f"image{i}_name"].strip()
                              for i in range(1, 6)
                              if f"image{i}_name" in row and row[f"image{i}_name"].strip()]
                    if not images:
                        continue
                    try:
                        label = images.index(correct_image)
                    except ValueError:
                        continue
                    # Read REAL captions from Extended TSV (same schema as Train/Dev)
                    captions = [row.get(f"image{i}_caption", "").strip()
                                for i in range(1, len(images) + 1)]
                    for idiom_folder in [idiom, idiom.replace("'", "_"), idiom.replace("'", "")]:
                        if os.path.exists(os.path.join(EXTENDED_IMG, idiom_folder, images[0])):
                            break
                    full_paths = [os.path.join(EXTENDED_IMG, idiom_folder, img) for img in images]
                    train_data.append({"idiom": idiom, "sentence": sentence,
                                       "images": full_paths, "captions": captions, "label": label})
                    added += 1
            print(f"  Extended: added {added} new idioms (with real captions) → total train = {len(train_data)}")

        print(f"  Task A Train: {len(train_data)} unique idioms")
        print(f"  Task A Val:   {len(val_data)} unique idioms")

        train_cache = cache_split_phase3(train_data, "Task A Train", clip_model, preprocess, cn_embeddings, device)
        val_cache   = cache_split_phase3(val_data,   "Task A Val",   clip_model, preprocess, cn_embeddings, device)

        torch.save(train_cache, p3_a_train)
        torch.save(val_cache,   p3_a_val)
        print(f"\n  ✓ Phase 3 Task A cached: {p3_a_train}")
    else:
        print(f"\n  ⚠ Augmented Task A TSV not found at {AUGMENTED_A_TRAIN}")

    # ── Task B ────────────────────────────────────────────────────────────────
    p3_b_train = os.path.join(CACHE_DIR, "phase3_task_b_train.pt")
    p3_b_val   = os.path.join(CACHE_DIR, "phase3_task_b_val.pt")

    if os.path.exists(p3_b_train):
        print("\nPhase 3 Task B cache already exists.")
    elif os.path.exists(AUGMENTED_B_TRAIN):
        print("\nProcessing Task B...")
        train_b = load_augmented_task_b(AUGMENTED_B_TRAIN, cfg.TASK_B_TRAIN_IMG)
        val_b   = load_augmented_task_b(AUGMENTED_B_VAL,   cfg.TASK_B_VAL_IMG)

        print(f"  Task B Train: {len(train_b)} unique idioms")
        print(f"  Task B Val:   {len(val_b)} unique idioms")

        train_b_cache = cache_split_phase3(train_b, "Task B Train", clip_model, preprocess, cn_embeddings, device)
        val_b_cache   = cache_split_phase3(val_b,   "Task B Val",   clip_model, preprocess, cn_embeddings, device)

        torch.save(train_b_cache, p3_b_train)
        torch.save(val_b_cache,   p3_b_val)
        print(f"\n  ✓ Phase 3 Task B cached: {p3_b_train}")

    print("\n" + "=" * 60)
    print("  Phase 3 caching complete!")
    print("  Next step: python evaluate_phase3.py --task a")
    print("=" * 60)


if __name__ == "__main__":
    main()
