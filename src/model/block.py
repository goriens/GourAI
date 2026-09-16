"""
Transformer Block module for GourAI V1.
Combines Pre-LayerNorm, Causal Multi-Head Self-Attention, and Feed-Forward MLP
with highway Residual Connections.
"""
import torch
import torch.nn as nn
from src.config import GourAIConfig
from src.model.attention import CausalSelfAttention
from src.model.mlp import MLP


class Block(nn.Module):
    def __init__(self, config: GourAIConfig):
        super().__init__()
        # Pre-LayerNorm 1 for Attention
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)

        # Pre-LayerNorm 2 for Feed-Forward MLP
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Formulation: x = x + SubLayer(LayerNorm(x))
        # 1. Communication step (Attention) with residual highway
        x = x + self.attn(self.ln_1(x))
        # 2. Computation step (MLP) with residual highway
        x = x + self.mlp(self.ln_2(x))
        return x


if __name__ == "__main__":
    from src.config import GourAIConfig

    cfg = GourAIConfig(n_embd=384, n_head=6, block_size=256)
    block = Block(cfg)

    # Dummy input: (B=4, T=8, C=384)
    dummy_x = torch.randn(4, 8, 384)
    out = block(dummy_x)

    print("================ GourAI Block Self-Verification ================")
    print(f"Input shape (B, T, C)    : {dummy_x.shape}")
    print(f"Output shape (B, T, C)   : {out.shape} (Expected: [4, 8, 384])")
    assert out.shape == (4, 8, 384), f"Expected (4, 8, 384), got {out.shape}"
    print("[SUCCESS] Transformer Block passed self-verification!")
    print("================================================================")
