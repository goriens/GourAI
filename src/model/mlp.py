"""
Feed-Forward Network (MLP) module for GourAI V1.
Implements the computation layer after multi-head attention:
Linear(4 * n_embd) -> GELU -> Linear(n_embd) -> Dropout.
"""
import torch
import torch.nn as nn
from src.config import GourAIConfig


class MLP(nn.Module):
    def __init__(self, config: GourAIConfig):
        super().__init__()
        # Standard Transformer 4x hidden expansion
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Expand dimension 4x (384 -> 1536)
        x = self.c_fc(x)
        # 2. Smooth non-linear activation
        x = self.gelu(x)
        # 3. Project back to embedding dimension (1536 -> 384)
        x = self.c_proj(x)
        # 4. Regularization dropout
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    from src.config import GourAIConfig

    cfg = GourAIConfig(n_embd=384)
    mlp = MLP(cfg)

    # Dummy input: (B=4, T=8, C=384)
    dummy_x = torch.randn(4, 8, 384)
    out = mlp(dummy_x)

    print("================ GourAI MLP Self-Verification ================")
    print(f"Input shape (B, T, C)    : {dummy_x.shape}")
    print(f"Output shape (B, T, C)   : {out.shape} (Expected: [4, 8, 384])")
    print(f"Hidden expansion         : {cfg.n_embd} -> {4 * cfg.n_embd} -> {cfg.n_embd}")
    assert out.shape == (4, 8, 384), f"Expected (4, 8, 384), got {out.shape}"
    print("[SUCCESS] MLP passed self-verification!")
    print("==============================================================")
