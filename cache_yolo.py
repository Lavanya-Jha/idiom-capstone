"""
cache_yolo.py
=============
PURPOSE:
    Phase 5 — YOLO Object Detection Cache.

    For each image in the dataset, run YOLO to detect objects and compute
    two new scalar signals that complement CLIP's holistic understanding:

    Signal 8: yolo_literal_score
        Measures whether objects detected by YOLO semantically match the
        idiom's LITERAL keywords (e.g., "bucket" for "kick the bucket").
        High score = literal objects found = likely a LITERAL interpretation image
        (which should rank LOWER in the figurative ranking task).

        Method: YOLO detects class labels → encode with CLIP text encoder →
        compute max cosine similarity with the idiom's literal CLIP embedding.
        This handles semantic overlap gracefully (e.g., YOLO detects "bowl",
        idiom is "bucket" — CLIP knows they are semantically similar).

    Signal 9: yolo_scene_richness
        Number of unique object classes detected / total detections.
        Figurative images tend to show complex, multi-object scenes.
        Literal images tend to be simpler, single-object shots.

HOW TO RUN:
    python cache_yolo.py

    Option A — Standard YOLOv8 (fast, 80 COCO classes):
        pip install ultralytics
        python cache_yolo.py --model yolov8n.pt

    Option B — YOLO-World (open-vocabulary, recommended):
        pip install ultralytics
        python cache_yolo.py --model yolov8x-worldv2.pt

OUTPUT:
    cache/yolo_task_a_train.pt
    cache/yolo_task_a_val.pt
    cache/yolo_task_b_train.pt  (if available)
    cache/yolo_task_b_val.pt    (if available)

EACH CACHE CONTAINS:
    yolo_literal_score:   (N, C)    — YOLO detection vs. literal idiom match
    yolo_scene_richness:  (N, C)    — object diversity per image
    idioms:               list[str]
    labels:               (N,)

NEXT STEP:
    python train_phase5.py --task a
"""

import os
import csv
import ast
import argparse
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

import config_phase1 as cfg

# ==============================================================================
# PATH RESOLUTION
# Scripts live in idiom-capstone\ and data is co-located in the same folder.
# We first check the script's own directory, then one level up (original design).
# ==============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve_data_root():
    """Find the folder that directly contains 'Subtask A'."""
    # Check script directory first (data is co-located with scripts)
    if os.path.isdir(os.path.join(_SCRIPT_DIR, "Subtask A")):
        return _SCRIPT_DIR
    # Fall back to one level up (original Capstone/phase1/ design)
    parent = os.path.dirname(_SCRIPT_DIR)
    if os.path.isdir(os.path.join(parent, "Subtask A")):
        return parent
    return _SCRIPT_DIR

_DATA_ROOT = _resolve_data_root()

# Plain TSV paths (confirmed from actual folder exploration)
TASK_A_TRAIN_TSV = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Train", "train", "subtask_a_train.tsv")
TASK_A_VAL_TSV   = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Dev",   "subtask_a_dev.tsv")
TASK_B_TRAIN_TSV = os.path.join(_DATA_ROOT, "Subtask B", "Train", "subtask_b_train.tsv")
TASK_B_VAL_TSV   = os.path.join(_DATA_ROOT, "Subtask B", "Dev",   "subtask_b_dev.tsv")

# Image root folders
TASK_A_TRAIN_IMG = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Train", "train")
TASK_A_VAL_IMG   = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Dev")
TASK_B_TRAIN_IMG = os.path.join(_DATA_ROOT, "Subtask B", "Train")
TASK_B_VAL_IMG   = os.path.join(_DATA_ROOT, "Subtask B", "Dev")

# Cache output goes in the scripts directory (where models/ and cache/ already live)
CACHE_DIR = os.path.join(_SCRIPT_DIR, "cache")

# ── Optional YOLO-World custom vocabulary per idiom ──────────────────────────
# These are the LITERAL nouns to query for each idiom.
# YOLO-World will search for these specific objects.
# For standard YOLOv8 this is ignored (it uses fixed COCO classes).
IDIOM_LITERAL_VOCAB = {
    "kick the bucket":      ["bucket", "person kicking"],
    "over the moon":        ["moon", "person flying"],
    "elbow grease":         ["elbow", "grease", "scrubbing"],
    "bite the bullet":      ["bullet", "teeth", "biting"],
    "break a leg":          ["leg", "breaking"],
    "hit the nail on the head": ["nail", "hammer", "head"],
    "let the cat out of the bag": ["cat", "bag"],
    "under the weather":    ["rain", "weather", "sick person"],
    "spill the beans":      ["beans", "spilling", "bowl"],
    "the ball is in your court": ["ball", "court", "tennis"],
    "cost an arm and a leg": ["arm", "leg", "money"],
    "it's raining cats and dogs": ["cats", "dogs", "rain"],
    "bite off more than you can chew": ["food", "mouth", "chewing"],
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


# ==============================================================================
# DATA LOADING — reuse augmented TSV (same as Phase 3)
# ==============================================================================

def load_task_a(tsv_path, images_root):
    seen = set()
    data = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idiom = row["compound"].strip()
            if idiom in seen:
                continue
            seen.add(idiom)
            sentence = row.get("sentence", idiom).strip() or idiom
            try:
                expected_order = ast.literal_eval(row["expected_order"])
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
            for idiom_folder in [idiom, idiom.replace("'", "_"), idiom.replace("'", "")]:
                if os.path.exists(os.path.join(images_root, idiom_folder, images[0])):
                    break
            full_paths = [os.path.join(images_root, idiom_folder, img) for img in images]
            data.append({"idiom": idiom, "sentence": sentence,
                         "images": full_paths, "label": label})
    return data


def load_task_b(tsv_path, images_root):
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
            images = [row[f"image{i}_name"].strip()
                      for i in range(1, 8)
                      if f"image{i}_name" in row and row[f"image{i}_name"].strip()]
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
            data.append({"idiom": idiom, "images": full_paths, "label": label})
    return data


# ==============================================================================
# LITERAL KEYWORD EXTRACTION (fallback for idioms not in IDIOM_LITERAL_VOCAB)
# ==============================================================================

STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for",
             "with", "and", "or", "but", "is", "are", "was", "were",
             "your", "my", "our", "their", "its", "this", "that"}


def get_literal_keywords(idiom):
    """Extract content words from idiom as literal search queries."""
    if idiom.lower() in IDIOM_LITERAL_VOCAB:
        return IDIOM_LITERAL_VOCAB[idiom.lower()]
    # Fallback: use non-stopword nouns from the idiom
    words = [w.strip("'s,.'\"") for w in idiom.lower().split()]
    keywords = [w for w in words if w and w not in STOPWORDS and len(w) > 2]
    return keywords if keywords else [idiom]


# ==============================================================================
# YOLO DETECTION
# ==============================================================================

def load_yolo(model_name):
    """Load YOLO model."""
    from ultralytics import YOLO
    print(f"  Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    print(f"  YOLO loaded.")
    return model


def detect_objects(yolo_model, image_path, idiom=None, conf_thresh=0.25):
    """
    Run YOLO on one image.

    For YOLO-World: set custom classes from idiom-specific vocabulary.
    For standard YOLOv8: use fixed COCO 80 classes.

    Returns:
        detected: list of (class_name, confidence) tuples
    """
    try:
        # If YOLO-World and idiom provided, set custom vocabulary
        if idiom and hasattr(yolo_model, 'set_classes'):
            keywords = get_literal_keywords(idiom)
            # Also add broader context words for richer scene description
            broader = keywords + ["person", "car", "animal", "building", "food"]
            yolo_model.set_classes(broader)

        results = yolo_model(image_path, conf=conf_thresh, verbose=False)
        detected = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf   = float(box.conf.item())
                name   = yolo_model.names[cls_id] if hasattr(yolo_model, 'names') else str(cls_id)
                detected.append((name, conf))
        return detected
    except Exception as e:
        return []


# ==============================================================================
# SIGNAL COMPUTATION
# ==============================================================================

def compute_yolo_signals(detected_objects, literal_embed, clip_model, device):
    """
    Compute two scalar signals from YOLO detections.

    Args:
        detected_objects: list of (class_name, confidence)
        literal_embed:    (512,) CLIP text embedding of literal idiom
        clip_model:       frozen CLIP model
        device:           torch device

    Returns:
        yolo_literal_score:  float — max CLIP similarity between detected objects and literal idiom text
        yolo_scene_richness: float — unique_classes / total_detections (0 if no detections)
    """
    if not detected_objects:
        return 0.0, 0.0

    # Signal 8: YOLO literal score
    # Encode each detected class name with CLIP and compare to literal embedding
    class_names = list({name for name, _ in detected_objects})  # unique names
    try:
        tokens = clip.tokenize([f"a photo of {n}" for n in class_names], truncate=True).to(device)
        with torch.no_grad():
            class_embeds = clip_model.encode_text(tokens)           # (K, 512)
            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
        sims = (class_embeds * literal_embed.unsqueeze(0)).sum(dim=-1)  # (K,)
        yolo_literal_score = float(sims.max().item())
    except Exception:
        yolo_literal_score = 0.0

    # Signal 9: Scene richness = unique classes / total detections
    total_dets  = len(detected_objects)
    unique_cls  = len(set(name for name, _ in detected_objects))
    yolo_scene_richness = unique_cls / total_dets if total_dets > 0 else 0.0

    return yolo_literal_score, yolo_scene_richness


# ==============================================================================
# CORE CACHING
# ==============================================================================

def cache_split_yolo(data, split_name, yolo_model, clip_model, device):
    """
    Run YOLO on every image in a split and compute YOLO-based signals.
    """
    literal_scores_all  = []
    richness_scores_all = []
    labels_all          = []
    idioms_all          = []
    skipped             = 0

    print(f"\n  Processing {split_name} ({len(data)} samples)...")

    for item in tqdm(data, desc=f"  {split_name}"):
        idiom    = item["idiom"]
        label    = int(item["label"])
        images   = item["images"]

        # Compute literal CLIP embedding for this idiom
        literal_text = f"a photo showing {' and '.join(get_literal_keywords(idiom)[:3])}"
        token = clip.tokenize([literal_text], truncate=True).to(device)
        with torch.no_grad():
            lit_embed = clip_model.encode_text(token)            # (1, 512)
            lit_embed = lit_embed / lit_embed.norm(dim=-1, keepdim=True)
        lit_embed = lit_embed.squeeze(0)                          # (512,)

        literal_scores  = []
        richness_scores = []

        for img_path in images:
            if not os.path.exists(img_path):
                literal_scores.append(0.0)
                richness_scores.append(0.0)
                skipped += 1
                continue

            # Run YOLO
            detected = detect_objects(yolo_model, img_path, idiom=idiom)

            # Compute signals
            lit_score, richness = compute_yolo_signals(detected, lit_embed, clip_model, device)
            literal_scores.append(lit_score)
            richness_scores.append(richness)

        literal_scores_all.append(torch.tensor(literal_scores, dtype=torch.float32))
        richness_scores_all.append(torch.tensor(richness_scores, dtype=torch.float32))
        labels_all.append(label)
        idioms_all.append(idiom)

    if skipped > 0:
        print(f"\n  {skipped} images skipped (not found)")

    return {
        "yolo_literal_score":   torch.stack(literal_scores_all,  dim=0).cpu(),  # (N, C)
        "yolo_scene_richness":  torch.stack(richness_scores_all, dim=0).cpu(),  # (N, C)
        "labels":               torch.tensor(labels_all, dtype=torch.long),
        "idioms":               idioms_all
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO model: yolov8n.pt (standard) or yolov8x-worldv2.pt (YOLO-World)")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("  Phase 5 — YOLO Object Detection Cache")
    print("=" * 60)
    print(f"  YOLO model: {args.model}")
    print("  New signals vs Phase 3:")
    print("    + yolo_literal_score  (YOLO detected objects vs idiom literal text)")
    print("    + yolo_scene_richness (unique objects / total detections)")

    # Load YOLO
    yolo_model = load_yolo(args.model)

    # Load CLIP (for computing CLIP similarity of YOLO labels)
    print(f"\nLoading CLIP {cfg.CLIP_MODEL}...")
    device = "cpu"
    clip_model, _ = clip.load(cfg.CLIP_MODEL, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    print("CLIP loaded.")

    print(f"\n  Data root: {_DATA_ROOT}")
    print(f"  Task A train TSV: {TASK_A_TRAIN_TSV}")
    print(f"  TSV found: {os.path.exists(TASK_A_TRAIN_TSV)}")

    # ── Task A ────────────────────────────────────────────────────────────────
    yolo_a_train = os.path.join(CACHE_DIR, "yolo_task_a_train.pt")
    yolo_a_val   = os.path.join(CACHE_DIR, "yolo_task_a_val.pt")

    if os.path.exists(yolo_a_train):
        print("\nYOLO Task A cache already exists. Delete to rebuild.")
    elif os.path.exists(TASK_A_TRAIN_TSV):
        print("\nProcessing Task A...")
        train_data = load_task_a(TASK_A_TRAIN_TSV, TASK_A_TRAIN_IMG)
        val_data   = load_task_a(TASK_A_VAL_TSV,   TASK_A_VAL_IMG)

        # Also include Extended set — Phase 3 does the same, so we must match
        # its 100 training samples exactly (70 train + 30 extended).
        extended_tsv = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Extended", "xeval", "subtask_a_xe.tsv")
        extended_img = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Extended", "xeval")
        if os.path.exists(extended_tsv):
            seen_idioms = {item["idiom"] for item in train_data}
            ext_data = load_task_a(extended_tsv, extended_img)
            added = 0
            for item in ext_data:
                if item["idiom"] not in seen_idioms:
                    train_data.append(item)
                    seen_idioms.add(item["idiom"])
                    added += 1
            print(f"  Extended: added {added} idioms → total train = {len(train_data)}")
        else:
            print(f"  (Extended TSV not found — train will have {len(train_data)} idioms)")

        print(f"  Task A Train: {len(train_data)} idioms")
        print(f"  Task A Val:   {len(val_data)} idioms")

        train_cache = cache_split_yolo(train_data, "Task A Train", yolo_model, clip_model, device)
        val_cache   = cache_split_yolo(val_data,   "Task A Val",   yolo_model, clip_model, device)
        torch.save(train_cache, yolo_a_train)
        torch.save(val_cache,   yolo_a_val)
        print(f"\n  YOLO Task A cached: {yolo_a_train}")
    else:
        print(f"\n  Task A TSV not found at {TASK_A_TRAIN_TSV}")
        print(f"  Check that 'Subtask A' exists inside: {_DATA_ROOT}")

    # ── Task B ────────────────────────────────────────────────────────────────
    yolo_b_train = os.path.join(CACHE_DIR, "yolo_task_b_train.pt")
    yolo_b_val   = os.path.join(CACHE_DIR, "yolo_task_b_val.pt")

    if os.path.exists(yolo_b_train):
        print("\nYOLO Task B cache already exists.")
    elif os.path.exists(TASK_B_TRAIN_TSV):
        print("\nProcessing Task B...")
        train_b = load_task_b(TASK_B_TRAIN_TSV, TASK_B_TRAIN_IMG)
        val_b   = load_task_b(TASK_B_VAL_TSV,   TASK_B_VAL_IMG)
        train_b_cache = cache_split_yolo(train_b, "Task B Train", yolo_model, clip_model, device)
        val_b_cache   = cache_split_yolo(val_b,   "Task B Val",   yolo_model, clip_model, device)
        torch.save(train_b_cache, yolo_b_train)
        torch.save(val_b_cache,   yolo_b_val)
        print(f"\n  YOLO Task B cached: {yolo_b_train}")

    print("\n" + "=" * 60)
    print("  YOLO caching complete!")
    print("  Next step: python train_phase5.py --task a")
    print("=" * 60)


if __name__ == "__main__":
    main()
