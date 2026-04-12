# Visual Grounding and Commonsense Reasoning for Multimodal Figurative Language Understanding

**SemEval-2025 Task — AdMIRe Dataset**

A multimodal system that ranks images by figurative relevance to idiomatic sentences using CLIP zero-shot similarity, idiom-aware prompt decomposition (IAPD), ConceptNet commonsense reasoning, and a trained Phase 4 Caption-Fused Ranker MLP.

---

## Architecture

```
Sentence ──► Idiom Lookup (IDIOM_DB, 147 idioms)
                │
                ▼
         IAPD Prompts ──► [Literal | Figurative | Contextual (de-idiomised)]
                │
                ▼
         CLIP ViT-B/32 ──► Image Embeddings
                │
                ▼
         FLGS Scoring ──► ctx_score + gap_weight × (fig_score − lit_score)
                │
                ▼
         Phase 4 MLP ──► CaptionFusedRanker (8 signals, 201 params)
                │
                ▼
         Auto-Categorise ──► [Figurative | Literal | Partial Literal | Random | Distractor]
                │
                ▼
         Visual Commonsense Chain ──► 7-section structured reasoning output
```

---

## Results

| Model | Task A Val | Task B Val |
|---|---|---|
| CLIP zero-shot (Phase 2 baseline) | 66.7% | 60.0% |
| Phase 4 MLP CaptionFused | **73.3%** | **80.0%** |
| Improvement | +6.6% | +20.0% |

---

## Key Components

| File | Description |
|---|---|
| `app.py` | Flask demo server — full pipeline + frontend |
| `phase4_model.py` | CaptionFusedRanker MLP (8 signals → ranking score) |
| `train_phase4.py` | Training script — pairwise MarginRankingLoss |
| `evaluate_phase4.py` | Ablation study — zero-shot per-signal vs MLP |
| `cache_phase3.py` | Pre-compute CLIP embeddings for Extended dataset |
| `setup_conceptnet.py` | Download and pickle ConceptNet NumberBatch |
| `requirements_phase1.txt` | Python dependencies |

---

## Setup

### 1. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements_phase1.txt
```

### 2. Download ConceptNet NumberBatch
```bash
python setup_conceptnet.py
```
This downloads the English NumberBatch embeddings (~900 MB) into `conceptnet/`.

### 3. Pre-compute caches
```bash
python cache_phase3.py          # CLIP embeddings for Extended dataset
```

### 4. Train Phase 4
```bash
python train_phase4.py --task a
python train_phase4.py --task b
```
Checkpoints saved to `models/phase4_task_a_mlp.pt` and `models/phase4_task_b_mlp.pt`.

### 5. Run the demo
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## Visual Commonsense Reasoning Chain

Each analysis produces a 7-section structured output:

1. **Visual Evidence** — CLIP zero-shot object & scene detection
2. **Literal Meaning** — physical scene description + fig−lit similarity gap
3. **Commonsense Inference** — scene implication + ConceptNet curated word associations
4. **Best Idiom** — matched idiom, meaning, origin, de-idiomised probe
5. **Why It Fits** — full score breakdown (fig/lit/ctx/gap/FLGS + visual overlap)
6. **Other Candidates Rejected** — top 3 near-miss idioms with rejection rationale
7. **Final Figurative Meaning** — conclusion: idiom + category + confidence

---

## Dataset

This project uses the **AdMIRe / SemEval-2025 Task 1** dataset.
Images are not included in this repository (not redistributable).
See the official dataset page for access.

---

## Technical Details

- **CLIP model**: `ViT-B/32` (OpenAI)
- **ConceptNet**: NumberBatch 300-dim English embeddings
- **IAPD de-idiomisation**: idiom phrase replaced with its meaning in contextual prompt to prevent CLIP anchoring to literal words
- **FLGS formula**: `ctx_score + gap_weight × (fig_score − lit_score)` where `gap_weight = 0.40` (known idiom) or `0.15` (unknown)
- **Phase 4 blend**: `0.6 × P4_norm + 0.4 × FLGS_norm`
- **Idiom DB**: 147 idioms with meaning, figurative visual prompt, literal visual prompt, origin
