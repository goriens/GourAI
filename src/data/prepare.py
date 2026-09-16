"""
Data preparation script for GourAI V1.
Downloads Tiny Shakespeare (~1MB) dataset, builds vocabulary mapping,
splits into 90% train and 10% validation, and serializes into binary files.
"""
import os
import json
import urllib.request
from pathlib import Path
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare_data(data_dir: str = "data"):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    input_file_path = data_path / "input.txt"

    # Step 1: Download dataset if not already present
    if not input_file_path.exists():
        print(f"[DOWNLOAD] Downloading Tiny Shakespeare from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, input_file_path)
        print(f"[SUCCESS] Downloaded to {input_file_path}")
    else:
        print(f"[INFO] Found existing dataset at {input_file_path}")

    # Step 2: Read raw text
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()

    n = len(data)
    print(f"[STATS] Total characters in dataset: {n:,}")

    # Step 3: Extract unique characters and create vocabulary
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"[VOCAB] Unique characters (vocab_size): {vocab_size}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Save vocabulary mapping for inference & tokenizer
    checkpoints_path = Path("checkpoints")
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    vocab_meta = {
        "vocab_size": vocab_size,
        "chars": chars,
        "stoi": stoi,
        "itos": itos,
    }
    with open(checkpoints_path / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_meta, f, indent=2)
    print(f"[SAVED] Vocabulary metadata saved to checkpoints/vocab.json")

    # Step 4: 90% Train / 10% Val split
    split_idx = int(n * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"[SPLIT] Train split: {len(train_data):,} chars (90%) | Val split: {len(val_data):,} chars (10%)")

    # Step 5: Encode text into integer sequences
    train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)

    # Step 6: Save as binary files for high-speed memory-mapped data loading
    train_bin_path = data_path / "train.bin"
    val_bin_path = data_path / "val.bin"

    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)

    print(f"[OUTPUT] Serialized binary files:")
    print(f"   - {train_bin_path} ({os.path.getsize(train_bin_path):,} bytes)")
    print(f"   - {val_bin_path} ({os.path.getsize(val_bin_path):,} bytes)")
    print("[DONE] Phase 2 Data Pipeline complete!")


if __name__ == "__main__":
    prepare_data()
