"""
Central configuration system for GourAI V1.
Contains model architecture parameters, training hyperparameters, and filesystem paths.
"""
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path


@dataclass
class GourAIConfig:
    # Architecture Hyperparameters
    block_size: int = 256        # Maximum context window (past tokens)
    vocab_size: int = 65         # Vocabulary size (65 for Shakespeare char, 4096 for BPE)
    n_layer: int = 6             # Number of Transformer blocks
    n_head: int = 6              # Number of Attention heads
    n_embd: int = 384            # Embedding / hidden representation dimension
    dropout: float = 0.2         # Dropout probability for regularization
    bias: bool = False           # True: bias in Linears and LayerNorms (like GPT-2). False: faster & cleaner (nanoGPT)

    # Training & Optimization Hyperparameters
    batch_size: int = 16         # Batch size (B) for CPU/light GPU
    learning_rate: float = 3e-4  # Maximum learning rate (AdamW)
    max_iters: int = 2000        # Total training steps
    warmup_iters: int = 100      # Linear warmup steps
    lr_decay_iters: int = 2000   # Cosine decay target steps
    min_lr: float = 3e-5         # Minimum learning rate after decay
    weight_decay: float = 0.1    # L2 regularization weight decay
    grad_clip: float = 1.0       # Gradient norm clipping threshold
    beta1: float = 0.9           # AdamW beta1
    beta2: float = 0.95          # AdamW beta2
    
    # Evaluation & Checkpointing
    eval_interval: int = 100     # Evaluate validation loss every N steps
    eval_iters: int = 20         # Number of batches to average loss over
    seed: int = 1337

    # Filesystem Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    best_model_name: str = "best_model.pt"

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GourAIConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "GourAIConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


if __name__ == "__main__":
    cfg = GourAIConfig()
    print("================ GourAI V1 Architecture Config ================")
    print(f"  Context Window (block_size): {cfg.block_size}")
    print(f"  Layers (n_layer):           {cfg.n_layer}")
    print(f"  Heads (n_head):             {cfg.n_head} (head_dim: {cfg.head_dim})")
    print(f"  Embedding Dim (n_embd):     {cfg.n_embd}")
    print(f"  Vocab Size:                 {cfg.vocab_size}")
    print(f"  Total Approx Parameters:    ~10.8 Million")
    print("===============================================================")
