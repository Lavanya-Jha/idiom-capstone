"""
setup_conceptnet.py
===================
PURPOSE:
    Download ConceptNet NumberBatch mini embeddings (English only).
    Run this ONCE before cache_phase2.py.

WHAT IS NUMBERBATCH?
    NumberBatch is a set of pre-computed 300-dim word vectors built from
    ConceptNet's knowledge graph. Unlike Word2Vec or GloVe, NumberBatch
    encodes commonsense meaning — so "bucket" is close to "container",
    "kick" is close to "strike", and "death" is close to "dying".

    For figurative language, this is powerful:
    - "kick the bucket" → bucket(container) + kick(strike) → death(dying)
    - The vector bridge between literal words and figurative meanings

WHY MINI (ENGLISH ONLY)?
    Full NumberBatch: ~600 MB (all languages)
    Mini English only: ~150 MB
    We only need English for this project.

HOW TO RUN:
    python setup_conceptnet.py

OUTPUT:
    conceptnet/numberbatch-en.h5   (downloaded)
    conceptnet/numberbatch_en.pkl  (fast-loading pickle cache)
"""

import os
import sys
import pickle
import urllib.request

CONCEPTNET_DIR  = os.path.join(os.path.dirname(__file__), "conceptnet")
H5_PATH         = os.path.join(CONCEPTNET_DIR, "numberbatch-en.h5")
PKL_PATH        = os.path.join(CONCEPTNET_DIR, "numberbatch_en.pkl")

# English-only mini file (~150 MB)
NUMBERBATCH_URL = (
    "https://conceptnet.s3.amazonaws.com/downloads/2019/"
    "numberbatch/numberbatch-en-19.08.txt.gz"
)


def download_with_progress(url, dest_path):
    """Download file with a simple progress display."""

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb  = downloaded / 1024 / 1024
            sys.stdout.write(f"\r  Downloading... {pct:.1f}%  ({mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook)
    print()  # newline after progress


def load_and_convert():
    """
    Load NumberBatch from text.gz format and save as pickle for fast reloading.

    NumberBatch text format:
        Line 1: <vocab_size> <dim>
        Line 2+: <word> <300 space-separated floats>
    """
    import gzip
    import numpy as np

    print("\n  Converting to fast-load format...")
    embeddings = {}

    gz_path = H5_PATH.replace(".h5", ".txt.gz")

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        first_line = f.readline().strip().split()
        vocab_size, dim = int(first_line[0]), int(first_line[1])
        print(f"  Vocabulary: {vocab_size:,} words, {dim}-dim vectors")

        for i, line in enumerate(f):
            parts = line.strip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec

            if (i + 1) % 50000 == 0:
                sys.stdout.write(f"\r  Loaded {i+1:,} / {vocab_size:,} words")
                sys.stdout.flush()

    print(f"\r  Loaded {len(embeddings):,} words total")

    # Save as pickle for instant loading next time
    with open(PKL_PATH, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  ✓ Saved to {PKL_PATH}")
    return embeddings


def main():
    os.makedirs(CONCEPTNET_DIR, exist_ok=True)

    print("=" * 60)
    print("  ConceptNet NumberBatch Setup")
    print("=" * 60)

    gz_path = H5_PATH.replace(".h5", ".txt.gz")

    if os.path.exists(PKL_PATH):
        print(f"\n✓ NumberBatch already set up at: {PKL_PATH}")
        print("  Nothing to do. Run cache_phase2.py next.")
        return

    # Download if not already present
    if not os.path.exists(gz_path):
        print(f"\nDownloading English NumberBatch (~150 MB)...")
        print(f"Source: {NUMBERBATCH_URL}\n")
        try:
            download_with_progress(NUMBERBATCH_URL, gz_path)
            print(f"\n✓ Downloaded to {gz_path}")
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            print("\nManual download instructions:")
            print(f"  1. Go to: https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/")
            print(f"  2. Download: numberbatch-en-19.08.txt.gz")
            print(f"  3. Place it at: {gz_path}")
            return
    else:
        print(f"\n✓ File already downloaded: {gz_path}")

    # Convert to pickle
    load_and_convert()

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("  Next step: python cache_phase2.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
