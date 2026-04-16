"""
cache_attention.py
==================
PURPOSE:
    Compute two novel spatially-grounded signals for figurative image ranking:

    Signal A: spatial_concentration  (N, C)
        Split each image into a 2x2 grid (4 quadrant crops). Encode each
        crop with CLIP and compute the STD of the 4 similarity scores to
        the figurative text embedding.

        HIGH std = one spatial region dominates = CLIP focuses on a single
                   object = literal interpretation image (bad rank)
        LOW  std = all regions contribute equally = scene-level figurative
                   meaning distributed across the whole image (good rank)

        This is spatially-grounded object detection: it identifies WHERE
        in the image the relevant semantics lie, using CLIP's own
        understanding rather than bounding boxes.

    Signal B: concept_gap  (N, C)
        Detect abstract concepts present IN the image using the curated
        INFERENCE_VOCAB (150 words from app.py):
          1. Find top-K vocab words most similar to the image embedding
             (these are the "concepts detected in the image")
          2. Compare detected concepts against figurative vs literal embeddings:
             gap = mean_sim(detected_concepts, fig) - mean_sim(detected_concepts, lit)

        Positive gap = detected image concepts align with figurative meaning
        Negative gap = detected image concepts align with literal words

        Why this fixes the previous version: instead of comparing text→text
        (fig_prompt vs lit_prompt, which are very similar in CLIP space),
        we go image→concepts→fig/lit, which uses the image as the bridge
        and produces much larger, more discriminative variance.

HOW TO RUN:
    python cache_attention.py

OUTPUT (overwrites previous attention cache):
    cache/attention_task_a_train.pt
    cache/attention_task_a_val.pt
    cache/attention_task_b_train.pt
    cache/attention_task_b_val.pt

EACH CACHE CONTAINS:
    spatial_concentration: (N, C)  — std of quadrant similarities to figurative text
    concept_gap:           (N, C)  — figurative vs literal concept match via image
    labels:                (N,)
    idioms:                list[str]

NEXT STEP:
    python train_phase6.py --task a
"""

import os
import csv
import ast
import torch
import clip
from PIL import Image
from tqdm import tqdm

import config_phase1 as cfg

# ── Path resolution ────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve_data_root():
    if os.path.isdir(os.path.join(_SCRIPT_DIR, "Subtask A")):
        return _SCRIPT_DIR
    parent = os.path.dirname(_SCRIPT_DIR)
    if os.path.isdir(os.path.join(parent, "Subtask A")):
        return parent
    return _SCRIPT_DIR

_DATA_ROOT = _resolve_data_root()
CACHE_DIR  = os.path.join(_SCRIPT_DIR, "cache")

TASK_A_TRAIN_TSV = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Train", "train", "subtask_a_train.tsv")
TASK_A_VAL_TSV   = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Dev",   "subtask_a_dev.tsv")
TASK_B_TRAIN_TSV = os.path.join(_DATA_ROOT, "Subtask B", "Train", "subtask_b_train.tsv")
TASK_B_VAL_TSV   = os.path.join(_DATA_ROOT, "Subtask B", "Dev",   "subtask_b_dev.tsv")

TASK_A_TRAIN_IMG = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Train", "train")
TASK_A_VAL_IMG   = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Dev")
TASK_B_TRAIN_IMG = os.path.join(_DATA_ROOT, "Subtask B", "Train")
TASK_B_VAL_IMG   = os.path.join(_DATA_ROOT, "Subtask B", "Dev")


# ── Curated inference vocabulary (from app.py) ────────────────────────────────

INFERENCE_VOCAB = [
    "anger","joy","fear","sadness","disgust","trust","surprise","anxiety",
    "stress","happiness","grief","shame","pride","guilt","love","hate",
    "envy","jealousy","hope","despair","confidence","excitement","frustration",
    "relief","disappointment","loneliness","regret","satisfaction","admiration",
    "corrupt","honest","dishonest","lazy","hardworking","clever","foolish",
    "stubborn","generous","greedy","brave","cowardly","loyal","disloyal",
    "deceptive","trustworthy","selfish","kind","cruel","innocent","naive",
    "wise","reckless","cautious","ambitious","humble","arrogant","patient",
    "impulsive","responsible","irresponsible","reliable","unreliable",
    "conflict","cooperation","betrayal","loyalty","rivalry","friendship",
    "leadership","authority","power","weakness","competition","collaboration",
    "manipulation","control","exclusion","inclusion","rejection","acceptance",
    "isolation","influence","dominance","submission","alliance","opposition",
    "struggle","compete","fight","escape","hide","reveal","succeed","fail",
    "help","harm","deceive","protect","attack","reward","punish","sacrifice",
    "risk","invest","waste","gain","lose","exploit","abuse","support","resist",
    "endure","overcome","abandon","pursue","achieve","avoid","confront",
    "success","failure","problem","solution","opportunity","threat","change",
    "growth","decay","progress","obstacle","consequence","responsibility",
    "freedom","constraint","advantage","disadvantage","burden","challenge",
    "pressure","crisis","danger","safety","loss","damage","benefit","reward",
    "justice","injustice","fairness","corruption","honesty","deception",
    "harm","good","evil","moral","unethical","punishment","integrity","sin",
    "team","group","individual","leader","follower","outsider","community",
    "society","family","relationship","partner","enemy","ally","stranger",
    "colleague","superior","subordinate","rival","victim","aggressor",
    "symbol","implication","meaning","expression","hidden","literal","irony",
    "metaphor","representation","indication","suggestion","inference",
]

STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for",
             "with", "and", "or", "but", "is", "are", "was", "were",
             "your", "my", "its", "this", "that", "very", "so"}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_task_a(tsv_path, images_root):
    seen, data = set(), []
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
            for folder in [idiom, idiom.replace("'", "_"), idiom.replace("'", "")]:
                if os.path.exists(os.path.join(images_root, folder, images[0])):
                    break
            full_paths = [os.path.join(images_root, folder, img) for img in images]
            data.append({"idiom": idiom, "sentence": sentence,
                         "images": full_paths, "label": label})
    return data


def load_task_b(tsv_path, images_root):
    seen, data = set(), []
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
            for folder in [idiom, idiom.replace("'", "_"), idiom.replace("'", "")]:
                if os.path.exists(os.path.join(images_root, folder, images[0])):
                    break
            full_paths = [os.path.join(images_root, folder, img) for img in images]
            data.append({"idiom": idiom, "images": full_paths, "label": label})
    return data


# ==============================================================================
# VOCABULARY PRE-ENCODING
# ==============================================================================

def build_vocab_embeds(clip_model, device):
    """
    Pre-encode all INFERENCE_VOCAB words with CLIP text encoder.
    Returns: (V, 512) normalised text embeddings.
    """
    prompts    = [f"a photo that shows {w}" for w in INFERENCE_VOCAB]
    all_embeds = []
    batch_size = 64
    for i in range(0, len(prompts), batch_size):
        tokens = clip.tokenize(prompts[i:i+batch_size], truncate=True).to(device)
        with torch.no_grad():
            feats = clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeds.append(feats.cpu())
    return torch.cat(all_embeds, dim=0)   # (V, 512)


# ==============================================================================
# SIGNAL A — SPATIAL CONCENTRATION
# ==============================================================================

def compute_spatial_concentration(clip_model, preprocess, image_pil, fig_embed, device):
    """
    Split image into a 2x2 grid (4 quadrant crops) and compute the STD of
    each crop's cosine similarity to the figurative text embedding.

    HIGH std = one quadrant dominates = single object present = LITERAL
    LOW  std = all quadrants similar  = whole scene is relevant = FIGURATIVE

    Args:
        image_pil:  PIL Image (full image)
        fig_embed:  (512,) normalised CLIP text embedding of figurative prompt

    Returns:
        std of the 4 quadrant similarities (scalar float)
    """
    W, H = image_pil.size

    quadrants = [
        image_pil.crop((0,    0,    W//2, H//2)),   # top-left
        image_pil.crop((W//2, 0,    W,    H//2)),   # top-right
        image_pil.crop((0,    H//2, W//2, H)),      # bottom-left
        image_pil.crop((W//2, H//2, W,    H)),      # bottom-right
    ]

    sims = []
    for q in quadrants:
        t = preprocess(q).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        sim = (feat.squeeze(0).cpu() * fig_embed).sum().item()
        sims.append(sim)

    return torch.tensor(sims).std().item()


# ==============================================================================
# SIGNAL B — IMAGE CONCEPT GAP
# ==============================================================================

def compute_concept_gap(img_embed, fig_embed, lit_embed, vocab_embeds, K=10):
    """
    Detect abstract concepts present in the image via CLIP vocabulary similarity,
    then measure whether those concepts lean figurative or literal.

    Steps:
      1. Find top-K INFERENCE_VOCAB words most similar to the image embedding.
         These are the "concepts detected in the image."
      2. Compute mean similarity of detected concept embeddings to fig_embed.
      3. Compute mean similarity of detected concept embeddings to lit_embed.
      4. gap = fig_sim - lit_sim

    Positive gap = image contains concepts aligning with figurative meaning.
    Negative gap = image contains concepts aligning with literal word meanings.

    Why this works better than text→text comparison:
      The image itself serves as the bridge — we find what abstract concepts
      are visible in it, then check which interpretation they support.

    Args:
        img_embed:    (512,) normalised CLIP image embedding
        fig_embed:    (512,) normalised CLIP figurative text embedding
        lit_embed:    (512,) normalised CLIP literal text embedding
        vocab_embeds: (V, 512) normalised vocab word embeddings
        K:            number of top concepts to use

    Returns:
        concept_gap (scalar float)
    """
    # Step 1: top-K vocab words by image similarity
    img_vocab_sims = (vocab_embeds * img_embed.unsqueeze(0)).sum(dim=-1)  # (V,)
    topk_idx       = img_vocab_sims.topk(K).indices                        # (K,)
    detected       = vocab_embeds[topk_idx]                                # (K, 512)

    # Step 2-4: compare detected concept embeddings to fig vs lit
    sim_fig = (detected * fig_embed.unsqueeze(0)).sum(dim=-1).mean().item()
    sim_lit = (detected * lit_embed.unsqueeze(0)).sum(dim=-1).mean().item()

    return sim_fig - sim_lit


# ==============================================================================
# IAPD PROMPT GENERATION
# ==============================================================================

def get_iapd_prompts(idiom, sentence):
    words     = [w.strip("'s,.'\"") for w in idiom.lower().split()
                 if w.strip("'s,.'\"") not in STOPWORDS and len(w) > 2]
    lit_text  = f"a photo showing {' and '.join(words[:3])}" if words else f"a literal photo of {idiom}"
    fig_text  = f"an image representing the figurative meaning of the expression '{idiom}'"
    return fig_text, lit_text


# ==============================================================================
# CORE CACHING
# ==============================================================================

def cache_split(data, split_name, clip_model, preprocess, vocab_embeds, device):
    spatial_all = []
    gap_all     = []
    labels_all  = []
    idioms_all  = []
    skipped     = 0

    print(f"\n  Processing {split_name} ({len(data)} samples)...")

    for item in tqdm(data, desc=f"  {split_name}"):
        idiom    = item["idiom"]
        sentence = item.get("sentence", idiom)
        label    = int(item["label"])
        images   = item["images"]

        # Figurative + literal CLIP text embeddings for this idiom
        fig_text, lit_text = get_iapd_prompts(idiom, sentence)
        tokens = clip.tokenize([fig_text, lit_text], truncate=True).to(device)
        with torch.no_grad():
            text_feats = clip_model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        fig_embed = text_feats[0].cpu()   # (512,)
        lit_embed = text_feats[1].cpu()   # (512,)

        spatial_scores = []
        gap_scores     = []

        for img_path in images:
            if not os.path.exists(img_path):
                spatial_scores.append(0.0)
                gap_scores.append(0.0)
                skipped += 1
                continue
            try:
                image_pil  = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(image_pil).unsqueeze(0).to(device)

                # Image embedding (needed for concept gap)
                with torch.no_grad():
                    img_feat = clip_model.encode_image(img_tensor)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                img_embed = img_feat.squeeze(0).cpu()

                # Signal A: spatial concentration (std of quadrant sims to fig)
                spatial = compute_spatial_concentration(
                    clip_model, preprocess, image_pil, fig_embed, device
                )

                # Signal B: image concept gap
                gap = compute_concept_gap(
                    img_embed, fig_embed, lit_embed, vocab_embeds, K=10
                )

            except Exception as e:
                print(f"\n    Warning: {img_path} — {e}")
                spatial = 0.0
                gap     = 0.0
                skipped += 1

            spatial_scores.append(spatial)
            gap_scores.append(gap)

        spatial_all.append(torch.tensor(spatial_scores, dtype=torch.float32))
        gap_all.append(torch.tensor(gap_scores,         dtype=torch.float32))
        labels_all.append(label)
        idioms_all.append(idiom)

    if skipped > 0:
        print(f"\n  {skipped} images skipped")

    return {
        "spatial_concentration": torch.stack(spatial_all, dim=0).cpu(),  # (N, C)
        "concept_gap":           torch.stack(gap_all,     dim=0).cpu(),  # (N, C)
        "labels":                torch.tensor(labels_all, dtype=torch.long),
        "idioms":                idioms_all,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("  Spatial + Concept Detection Cache (v2)")
    print("=" * 60)
    print(f"  Data root: {_DATA_ROOT}")
    print("  Signals:")
    print("    + spatial_concentration  (std of quadrant CLIP sims — spatial grounding)")
    print("    + concept_gap            (image-detected concepts vs figurative/literal)")

    print(f"\nLoading CLIP {cfg.CLIP_MODEL}...")
    device      = "cpu"
    clip_model, preprocess = clip.load(cfg.CLIP_MODEL, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    print("CLIP loaded.")

    print("\nPre-encoding inference vocabulary...")
    vocab_embeds = build_vocab_embeds(clip_model, device)   # (V, 512)
    print(f"  {len(INFERENCE_VOCAB)} concept words encoded.")

    # Delete stale caches so they get rebuilt with new signals
    for fname in ["attention_task_a_train.pt", "attention_task_a_val.pt",
                  "attention_task_b_train.pt", "attention_task_b_val.pt"]:
        p = os.path.join(CACHE_DIR, fname)
        if os.path.exists(p):
            os.remove(p)
            print(f"  Deleted stale cache: {fname}")

    # ── Task A ────────────────────────────────────────────────────────────────
    att_a_train = os.path.join(CACHE_DIR, "attention_task_a_train.pt")
    att_a_val   = os.path.join(CACHE_DIR, "attention_task_a_val.pt")

    if os.path.exists(TASK_A_TRAIN_TSV):
        print("\nProcessing Task A...")
        train_data = load_task_a(TASK_A_TRAIN_TSV, TASK_A_TRAIN_IMG)
        val_data   = load_task_a(TASK_A_VAL_TSV,   TASK_A_VAL_IMG)

        # Include Extended set (to match Phase 3's 100 training samples)
        extended_tsv = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Extended", "xeval", "subtask_a_xe.tsv")
        extended_img = os.path.join(_DATA_ROOT, "Subtask A", "EN", "Extended", "xeval")
        if os.path.exists(extended_tsv):
            seen = {item["idiom"] for item in train_data}
            ext  = load_task_a(extended_tsv, extended_img)
            added = 0
            for item in ext:
                if item["idiom"] not in seen:
                    train_data.append(item)
                    seen.add(item["idiom"])
                    added += 1
            print(f"  Extended: added {added} idioms → total train = {len(train_data)}")

        print(f"  Task A Train: {len(train_data)} | Val: {len(val_data)}")

        train_cache = cache_split(train_data, "Task A Train", clip_model, preprocess, vocab_embeds, device)
        val_cache   = cache_split(val_data,   "Task A Val",   clip_model, preprocess, vocab_embeds, device)
        torch.save(train_cache, att_a_train)
        torch.save(val_cache,   att_a_val)
        print(f"\n  Saved: {att_a_train}")
    else:
        print(f"\n  Task A TSV not found: {TASK_A_TRAIN_TSV}")

    # ── Task B ────────────────────────────────────────────────────────────────
    att_b_train = os.path.join(CACHE_DIR, "attention_task_b_train.pt")
    att_b_val   = os.path.join(CACHE_DIR, "attention_task_b_val.pt")

    if os.path.exists(TASK_B_TRAIN_TSV):
        print("\nProcessing Task B...")
        train_b = load_task_b(TASK_B_TRAIN_TSV, TASK_B_TRAIN_IMG)
        val_b   = load_task_b(TASK_B_VAL_TSV,   TASK_B_VAL_IMG)
        print(f"  Task B Train: {len(train_b)} | Val: {len(val_b)}")

        train_b_cache = cache_split(train_b, "Task B Train", clip_model, preprocess, vocab_embeds, device)
        val_b_cache   = cache_split(val_b,   "Task B Val",   clip_model, preprocess, vocab_embeds, device)
        torch.save(train_b_cache, att_b_train)
        torch.save(val_b_cache,   att_b_val)
        print(f"\n  Saved: {att_b_train}")

    print("\n" + "=" * 60)
    print("  Caching complete!")
    print("  Next: python diagnose_signals.py  (verify signal quality)")
    print("  Then: python train_phase6.py --task a")
    print("=" * 60)


if __name__ == "__main__":
    main()
