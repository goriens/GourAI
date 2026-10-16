"""
High-throughput memory-mapped DataLoader for GourAI V1.
Loads binary encoded token datasets (train.bin, val.bin) via numpy.memmap
and slices random (B, T) batches of input sequences x and target sequences y (shifted by 1 token).
"""
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from src.config import GourAIConfig
from src.device import get_device


class GourAIDataLoader:
    def __init__(self, data_dir: str = "data", device: torch.device = None):
        self.data_dir = Path(data_dir)
        self.device = device if device is not None else get_device()
        
        train_path = self.data_dir / "train.bin"
        val_path = self.data_dir / "val.bin"

        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Binary datasets not found in '{data_dir}'. "
                "Please run `python -m src.data.prepare` first."
            )

        # Memory map binary datasets for instant access without RAM bloat
        self.train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
        self.val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    def get_batch(
        self, split: str = "train", batch_size: int = 16, block_size: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a random mini-batch of (B, T) chunks from specified split.
        Returns:
            x: (B, T) tensor of input token IDs
            y: (B, T) tensor of target token IDs (next token prediction)
        """
        data = self.train_data if split == "train" else self.val_data

        # Generate random start offsets ensuring full context window + 1 target token fits
        high = len(data) - block_size
        if high <= 0:
            raise ValueError(
                f"Dataset split '{split}' length ({len(data)}) is too small for block_size ({block_size})."
            )

        ix = torch.randint(0, high, (batch_size,))

        # Slice contiguous token sequences
        x_stacked = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
        )
        y_stacked = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )

        if self.device.type == "cuda":
            # Pin and async transfer to GPU
            x = x_stacked.pin_memory().to(self.device, non_blocking=True)
            y = y_stacked.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x_stacked.to(self.device)
            y = y_stacked.to(self.device)

        return x, y


# Standalone self-verification test
if __name__ == "__main__":
    print("================ GourAI DataLoader Verification ================")
    config = GourAIConfig()
    loader = GourAIDataLoader(data_dir=config.data_dir)

    print(f"Train tokens loaded : {len(loader.train_data):,}")
    print(f"Val tokens loaded   : {len(loader.val_data):,}")
    print(f"Sampling device     : {loader.device}")

    # Sample train batch
    x_train, y_train = loader.get_batch(
        split="train", batch_size=4, block_size=16
    )
    print(f"Sample Batch X shape: {x_train.shape} (Expected: [4, 16])")
    print(f"Sample Batch Y shape: {y_train.shape} (Expected: [4, 16])")

    # Verify shift by 1 property
    print("\nVerifying shift-by-1 property on batch index 0:")
    print(f"  x[0, :8] = {x_train[0, :8].tolist()}")
    print(f"  y[0, :8] = {y_train[0, :8].tolist()}")

    # Assert that y[0, :-1] matches x[0, 1:]
    assert torch.equal(
        x_train[0, 1:8], y_train[0, :7]
    ), "Target tokens are not properly shifted by 1!"
    print("  Shift assertion passed: y[t] == x[t+1]")

    # Sample val batch
    x_val, y_val = loader.get_batch(
        split="val", batch_size=4, block_size=16
    )
    assert x_val.shape == (4, 16) and y_val.shape == (4, 16)

    print("[SUCCESS] GourAI DataLoader passed all self-verification checks!")
    print("================================================================")
