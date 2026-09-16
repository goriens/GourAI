"""
Causal Multi-Head Self-Attention module for GourAI V1.
Computes attention scores between tokens using Query, Key, Value projections.
Supports PyTorch 2.0+ FlashAttention (scaled_dot_product_attention) with standard fallback.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.config import GourAIConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GourAIConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.block_size = config.block_size

        # Key, Query, Value projections in a single batched linear layer (3x expansion)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection layer
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix ensuring tokens only attend to past tokens
        # Register as buffer so it moves with model to GPU/CPU without being a trainable parameter
        self.register_buffer(
            "bias_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        # Check if PyTorch's native FlashAttention is available
        self.has_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # 1. Calculate Query, Key, Values for all heads in batch
        # shape: (B, T, 3 * n_embd) -> three tensors of (B, T, n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. Reshape into multi-head format: (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Causal self-attention
        if self.has_flash:
            # FlashAttention: Hardware accelerated, fast O(N) memory attention
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual Scaled Dot-Product Attention fallback
            # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Apply causal mask: future positions become -inf so softmax gives 0 probability
            att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # 4. Re-assemble all head outputs side-by-side: (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


if __name__ == "__main__":
    from src.config import GourAIConfig

    cfg = GourAIConfig(n_embd=384, n_head=6, block_size=256)
    attn = CausalSelfAttention(cfg)

    # Dummy input: batch of 4, sequence length 8, embedding dim 384
    dummy_x = torch.randn(4, 8, 384)
    out = attn(dummy_x)

    print("================ GourAI Attention Self-Verification ================")
    print(f"Input shape (B, T, C)    : {dummy_x.shape}")
    print(f"Output shape (B, T, C)   : {out.shape} (Expected: [4, 8, 384])")
    print(f"FlashAttention Enabled   : {attn.has_flash}")
    print(f"Heads                    : {attn.n_head} heads of dim {attn.head_dim}")
    assert out.shape == (4, 8, 384), f"Expected (4, 8, 384), got {out.shape}"
    print("[SUCCESS] CausalSelfAttention passed self-verification!")
    print("====================================================================")
