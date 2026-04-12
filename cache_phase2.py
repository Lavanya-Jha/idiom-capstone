"""
cache_phase2.py
===============
PURPOSE:
    Build Phase 2 embedding cache with three major improvements over Phase 1:

    1. IDIOM-AWARE PROMPT DECOMPOSITION (IAPD)
       Instead of 3 generic prompts, generate 3 semantically targeted perspectives:
         - Literal:     describes the actual objects/words in the idiom
         - Figurative:  describes the abstract/metaphorical meaning
         - Contextual:  the original sentence from the dataset
       Each perspective is encoded separately by CLIP → 3 × 512-dim embeddings.
       At inference, a learned attention layer weights them by idiom complexity.

    2. CONCEPTNET NUMBERBATCH INTEGRATION
       For each idiom, look up pre-computed 300-dim commonsense vectors for each
       word, then average them. This bridges the gap between literal word meanings
       (kick, bucket) and figurative meanings (death, end).
       Stored separately → fused in the Phase 2 model with a projection layer.

    3. EXTENDED DATASET INCLUSION
       Merges Task A Train (70) + Extended Evaluation (100) into one training split.
       - 30 new unique idioms from Extended → more coverage
       - Literal sentences from Extended → natural "literal perspective" for IAPD
       - Effective training set grows from 70 → 100 unique idioms

HOW TO RUN:
    python setup_conceptnet.py   (once — downloads ~150 MB)
    python cache_phase2.py

OUTPUT:
    cache/phase2_task_a_train.pt
    cache/phase2_task_a_val.pt
    cache/phase2_task_b_train.pt
    cache/phase2_task_b_val.pt
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

EXTENDED_A_TSV  = os.path.join(cfg.DATA_ROOT, "Subtask A", "EN", "Extended", "xeval", "subtask_a_xe.tsv")
EXTENDED_A_IMG  = os.path.join(cfg.DATA_ROOT, "Subtask A", "EN", "Extended", "xeval")

CONCEPTNET_PKL  = os.path.join(os.path.dirname(__file__), "conceptnet", "numberbatch_en.pkl")
CACHE_DIR_P2    = os.path.join(os.path.dirname(__file__), "cache")  # same cache folder, different filenames


# ==============================================================================
# IAPD — Idiom-Aware Prompt Decomposition
# ==============================================================================

def generate_iapd_prompts(idiom, sentence):
    """
    Generate 3 perspective-specific prompts for one idiom.

    WHY 3 PERSPECTIVES?
    Different idioms need different visual cues:
      - Simple idioms ("black sheep") → resolved at the object level (a black sheep)
      - Action idioms ("kick the bucket") → need action + figurative (kick + death)
      - Abstract idioms ("thin ice") → need full figurative context (danger/risk)

    By encoding all three perspectives and letting the model weight them,
    we cover all idiom types instead of forcing one description style.

    Args:
        idiom:    e.g. "kick the bucket"
        sentence: e.g. "He finally kicked the bucket after a long illness."

    Returns:
        dict with keys: 'literal', 'figurative', 'contextual'
    """
    words  = idiom.lower().split()

    # ── Literal Perspective ───────────────────────────────────────────────────
    # Describes the physical/visual objects in the idiom literally.
    # Helps ground abstract idioms in concrete visual elements.
    if len(words) >= 2:
        literal = f"a photo showing {' and '.join(words[:3])}"
    else:
        literal = f"a photo of {idiom}"

    # ── Figurative Perspective ────────────────────────────────────────────────
    # Describes the abstract/metaphorical meaning.
    # This is the most important perspective for figurative language tasks.
    figurative = f"an image representing the figurative meaning of the expression '{idiom}'"

    # ── Contextual Perspective ────────────────────────────────────────────────
    # Uses the original sentence from the dataset.
    # Provides real-world usage context that grounds the idiom semantically.
    contextual = sentence if sentence and sentence != idiom \
                 else f"a visual scene depicting '{idiom}'"

    return {
        "literal":    literal,
        "figurative": figurative,
        "contextual": contextual
    }


# ==============================================================================
# CONCEPTNET NUMBERBATCH
# ==============================================================================

def load_conceptnet(pkl_path):
    """Load pre-computed NumberBatch embeddings from pickle file."""
    if not os.path.exists(pkl_path):
        print(f"  ⚠ ConceptNet not found at {pkl_path}")
        print(f"    Run: python setup_conceptnet.py")
        return None

    print(f"  Loading ConceptNet NumberBatch...")
    with open(pkl_path, "rb") as f:
        embeddings = pickle.load(f)
    print(f"  ✓ Loaded {len(embeddings):,} word vectors")
    return embeddings


def get_conceptnet_vector(idiom, cn_embeddings):
    """
    Get a single 300-dim vector representing an idiom's commonsense meaning.

    Strategy:
        1. Split idiom into words (e.g. "kick the bucket" → [kick, the, bucket])
        2. Look up NumberBatch vector for each content word (skip stopwords)
        3. Average the vectors → single 300-dim representation

    NumberBatch keys use the format "/c/en/word" — we strip to just "word".

    Returns:
        numpy array (300,) — zero vector if no words found in vocabulary
    """
    stopwords = {"a", "an", "the", "of", "in", "on", "at", "to", "for",
                 "with", "and", "or", "but", "is", "are", "was", "were"}

    words = [w.lower().strip("'s") for w in idiom.split()
             if w.lower() not in stopwords]

    vecs = []
    for word in words:
        # NumberBatch stores words as plain lowercase strings
        if word in cn_embeddings:
            vecs.append(cn_embeddings[word])
        # Also try underscore form (e.g. "hot_air")
        elif idiom.replace(" ", "_").lower() in cn_embeddings:
            vecs.append(cn_embeddings[idiom.replace(" ", "_").lower()])

    if vecs:
        avg_vec = np.mean(vecs, axis=0).astype(np.float32)
        # L2 normalise
        norm = np.linalg.norm(avg_vec)
        if norm > 0:
            avg_vec = avg_vec / norm
        return avg_vec
    else:
        return np.zeros(300, dtype=np.float32)


# ==============================================================================
# DATA LOADING (same as Phase 1, reused)
# ==============================================================================

def load_task_a(tsv_path, images_root):
    """Load Task A TSV — same format as Phase 1."""
    data = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idiom    = row["compound"].strip()
            sentence = row.get("sentence", idiom).strip() or idiom
            try:
                expected_order = ast.literal_eval(row["expected_order"])
                correct_image  = expected_order[0]
            except Exception:
                continue

            images = []
            for i in range(1, 6):
                key = f"image{i}_name"
                if key in row and row[key].strip():
                    images.append(row[key].strip())

            if not images:
                continue

            try:
                label = images.index(correct_image)
            except ValueError:
                continue

            # Try multiple folder name variants (apostrophe kept, replaced with _, or removed)
            for idiom_folder in [idiom, idiom.replace("'", "_"), idiom.replace("'", "")]:
                if images and os.path.exists(os.path.join(images_root, idiom_folder, images[0])):
                    break

            full_paths   = [os.path.join(images_root, idiom_folder, img) for img in images]

            data.append({
                "idiom":    idiom,
                "sentence": sentence,
                "images":   full_paths,
                "label":    label
            })
    return data


def merge_and_deduplicate(train_data, extended_data):
    """
    Merge Train and Extended datasets, keeping unique idioms only.

    Strategy:
        - Keep all Train idioms (they are idiomatic sentences — better for figurative task)
        - Add Extended idioms NOT already in Train (30 new unique idioms)
        - For shared idioms, keep Train version (idiomatic sentence preferred)

    This gives 100 unique training idioms instead of 70.
    """
    seen_idioms  = {item["idiom"] for item in train_data}
    merged       = list(train_data)

    added = 0
    for item in extended_data:
        if item["idiom"] not in seen_idioms:
            merged.append(item)
            seen_idioms.add(item["idiom"])
            added += 1

    print(f"  Merged: {len(train_data)} train + {added} new from Extended = {len(merged)} total")
    return merged


# ==============================================================================
# CORE CACHING FUNCTION
# ==============================================================================

def cache_split_phase2(data, split_name, clip_model, preprocess, cn_embeddings, device):
    """
    Cache Phase 2 embeddings for one data split.

    Per sample, saves:
        iapd_embeddings:    (3, 512) — literal / figurative / contextual CLIP embeddings
        image_embeddings:   (num_candidates, 512) — frozen CLIP image embeddings
        conceptnet_vectors: (300,) — averaged NumberBatch vector for idiom words
        label:              int
        idiom:              str
    """
    iapd_all   = []
    image_all  = []
    cn_all     = []
    labels_all = []
    idioms_all = []
    skipped    = 0

    use_cn = cn_embeddings is not None
    print(f"\n  Processing {split_name} ({len(data)} samples)...")
    if not use_cn:
        print(f"  ⚠ ConceptNet not available — skipping CN vectors (won't affect IAPD)")

    for item in tqdm(data, desc=f"  {split_name}"):
        idiom    = item["idiom"]
        sentence = item.get("sentence", idiom)
        label    = int(item["label"])
        images   = item["images"]

        # ── IAPD: 3-perspective text embeddings ──────────────────────────────
        perspectives = generate_iapd_prompts(idiom, sentence)
        prompts      = [
            perspectives["literal"],
            perspectives["figurative"],
            perspectives["contextual"]
        ]

        text_tokens = clip.tokenize(prompts, truncate=True).to(device)  # (3, 77)
        with torch.no_grad():
            text_feats = clip_model.encode_text(text_tokens)            # (3, 512)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        iapd_embed = text_feats  # (3, 512)

        # ── ConceptNet vector ─────────────────────────────────────────────────
        if use_cn:
            cn_vec = get_conceptnet_vector(idiom, cn_embeddings)        # (300,)
        else:
            cn_vec = np.zeros(300, dtype=np.float32)

        # ── Image embeddings ──────────────────────────────────────────────────
        img_embeds = []
        for img_path in images:
            try:
                image      = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_feat = clip_model.encode_image(img_tensor)      # (1, 512)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                img_embeds.append(img_feat.squeeze(0))                   # (512,)
            except Exception as e:
                print(f"\n    ⚠ Could not load: {img_path} — {e}")
                img_embeds.append(torch.zeros(cfg.EMBED_DIM, device=device))
                skipped += 1

        image_embed = torch.stack(img_embeds, dim=0)  # (num_candidates, 512)

        iapd_all.append(iapd_embed)
        image_all.append(image_embed)
        cn_all.append(torch.tensor(cn_vec))
        labels_all.append(label)
        idioms_all.append(idiom)

    if skipped > 0:
        print(f"\n  ⚠ {skipped} images skipped — check apostrophe folder names")

    return {
        "iapd_embeddings":    torch.stack(iapd_all,  dim=0).cpu(),  # (N, 3, 512)
        "image_embeddings":   torch.stack(image_all, dim=0).cpu(),  # (N, C, 512)
        "conceptnet_vectors": torch.stack(cn_all,    dim=0).cpu(),  # (N, 300)
        "labels":             torch.tensor(labels_all, dtype=torch.long),
        "idioms":             idioms_all
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(CACHE_DIR_P2, exist_ok=True)

    print("=" * 60)
    print("  Phase 2 Embedding Cache Builder")
    print("=" * 60)

    # Load CLIP
    print(f"\nLoading CLIP {cfg.CLIP_MODEL} on CPU...")
    device = "cpu"
    clip_model, preprocess = clip.load(cfg.CLIP_MODEL, device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print("✓ CLIP loaded and frozen.\n")

    # Load ConceptNet
    cn_embeddings = load_conceptnet(CONCEPTNET_PKL)

    # ── Task A ────────────────────────────────────────────────────────────────
    p2_train_out = os.path.join(CACHE_DIR_P2, "phase2_task_a_train.pt")
    p2_val_out   = os.path.join(CACHE_DIR_P2, "phase2_task_a_val.pt")

    if os.path.exists(p2_train_out):
        print("Phase 2 Task A cache already exists. Delete cache/phase2_task_a_*.pt to rebuild.")
    else:
        print("Processing Task A (Train + Extended merge)...")

        train_data    = load_task_a(cfg.TASK_A_TRAIN, cfg.TASK_A_TRAIN_IMG)
        val_data      = load_task_a(cfg.TASK_A_VAL,   cfg.TASK_A_VAL_IMG)

        # Merge with Extended data
        if os.path.exists(EXTENDED_A_TSV):
            extended_data = load_task_a(EXTENDED_A_TSV, EXTENDED_A_IMG)
            train_data    = merge_and_deduplicate(train_data, extended_data)
        else:
            print(f"  ⚠ Extended TSV not found — using Train only")

        train_cache = cache_split_phase2(train_data, "Task A Train+Extended",
                                         clip_model, preprocess, cn_embeddings, device)
        val_cache   = cache_split_phase2(val_data,   "Task A Val",
                                         clip_model, preprocess, cn_embeddings, device)

        torch.save(train_cache, p2_train_out)
        torch.save(val_cache,   p2_val_out)

        print(f"\n  ✓ Phase 2 Task A cached:")
        print(f"    {p2_train_out}  ({len(train_data)} samples)")
        print(f"    {p2_val_out}    ({len(val_data)} samples)")

    # ── Task B ────────────────────────────────────────────────────────────────
    p2b_train_out = os.path.join(CACHE_DIR_P2, "phase2_task_b_train.pt")
    p2b_val_out   = os.path.join(CACHE_DIR_P2, "phase2_task_b_val.pt")

    if os.path.exists(cfg.TASK_B_TRAIN):
        print("\nProcessing Task B...")

        def load_task_b(tsv_path, images_root):
            data = []
            with open(tsv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    idiom         = row["compound"].strip()
                    correct_image = row["expected_item"].strip()
                    sentence      = row.get("sequence_caption1", idiom).strip() or idiom
                    images        = []
                    for i in range(1, 8):
                        key = f"image{i}_name"
                        if key in row and row[key].strip():
                            images.append(row[key].strip())
                    if not images:
                        continue
                    try:
                        label = images.index(correct_image)
                    except ValueError:
                        continue
                    # Try multiple folder name variants (apostrophe kept, replaced with _, or removed)
                    for idiom_folder in [idiom, idiom.replace("'", "_"), idiom.replace("'", "")]:
                        test_path = os.path.join(images_root, idiom_folder, images[0])
                        if os.path.exists(test_path):
                            break
                    full_paths   = [os.path.join(images_root, idiom_folder, img) for img in images]
                    data.append({"idiom": idiom, "sentence": sentence,
                                 "images": full_paths, "label": label})
            return data

        train_b = load_task_b(cfg.TASK_B_TRAIN, cfg.TASK_B_TRAIN_IMG)
        val_b   = load_task_b(cfg.TASK_B_VAL,   cfg.TASK_B_VAL_IMG)

        train_b_cache = cache_split_phase2(train_b, "Task B Train",
                                           clip_model, preprocess, cn_embeddings, device)
        val_b_cache   = cache_split_phase2(val_b,   "Task B Val",
                                           clip_model, preprocess, cn_embeddings, device)

        torch.save(train_b_cache, p2b_train_out)
        torch.save(val_b_cache,   p2b_val_out)

        print(f"\n  ✓ Phase 2 Task B cached:")
        print(f"    {p2b_train_out}  ({len(train_b)} samples)")
        print(f"    {p2b_val_out}    ({len(val_b)} samples)")

    print("\n" + "=" * 60)
    print("  Phase 2 caching complete!")
    print("  Next step: python train_phase2.py --task a")
    print("=" * 60)


if __name__ == "__main__":
    main()
