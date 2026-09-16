"""
Embeddings module for GourAI V1.
Combines Token Embeddings (wte) and learned Positional Embeddings (wpe).
"""
import torch
import torch.nn as nn
from src.config import GourAIConfig


class GourAIEmbeddings(nn.Module):
    def __init__(self, config: GourAIConfig):
        super().__init__()
        self.config = config

        # 1. Token Embeddings: maps each token ID to an n_embd dimensional vector
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # 2. Positional Embeddings: maps positions 0..block_size-1 to an n_embd dimensional vector
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # 3. Dropout for regularization
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for embeddings.
        Input: idx shape (B, T) where B is batch size, T is sequence length (T <= block_size)
        Output: shape (B, T, n_embd)
        """
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block_size is {self.config.block_size}"

        # Position indices: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Token embeddings + Positional embeddings
        tok_emb = self.wte(idx)       # shape: (B, T, n_embd)
        pos_emb = self.wpe(pos)       # shape: (T, n_embd)

        # Broadcasting automatically sums pos_emb across all batches B
        x = self.drop(tok_emb + pos_emb)  # shape: (B, T, n_embd)
        return x


if __name__ == "__main__":
    from src.config import GourAIConfig
    
    cfg = GourAIConfig(vocab_size=65, block_size=256, n_embd=384)
    emb_layer = GourAIEmbeddings(cfg)

    # Dummy batch: 4 sequences of length 8
    dummy_input = torch.randint(0, 65, (4, 8))
    out = emb_layer(dummy_input)

    print("================ GourAI Embeddings Self-Verification ================")
    print(f"Input shape (B, T)       : {dummy_input.shape}")
    print(f"Output shape (B, T, C)   : {out.shape} (Expected: [4, 8, 384])")
    assert out.shape == (4, 8, 384), f"Expected (4, 8, 384), got {out.shape}"
    print("[SUCCESS] Embeddings forward pass passed self-verification!")
    print("====================================================================")
