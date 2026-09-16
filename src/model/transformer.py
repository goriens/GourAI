"""
Master Transformer Language Model architecture for GourAI V1.
Stacks 6 Transformer blocks with Pre-LN, Final LayerNorm, and tied LM Head.
Supports forward pass with cross-entropy loss computation and parameter counting.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.config import GourAIConfig
from src.model.embeddings import GourAIEmbeddings
from src.model.block import Block


class GourAILM(nn.Module):
    def __init__(self, config: GourAIConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                emb=GourAIEmbeddings(config),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # Final projection to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: tie token embedding weights with output linear weights
        # According to 'Using the Output Embedding to Improve Language Models' (Press & Wolf 2016)
        self.transformer.emb.wte.weight = self.lm_head.weight

        # Initialize all model weights with Gaussian distribution
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count, the position embeddings get subtracted.
        The token embeddings would too, except due to weight tying it is also the LM head.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.emb.wpe.weight.numel()
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block_size is {self.config.block_size}"

        # 1. Forward through embeddings
        x = self.transformer.emb(idx)

        # 2. Forward through all N transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # 3. Final LayerNorm
        x = self.transformer.ln_f(x)

        # 4. If we have targets, compute loss; otherwise compute logits
        if targets is not None:
            # Full sequence forward pass during training
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference optimization: only forward the lm_head for the very last position
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss


if __name__ == "__main__":
    from src.config import GourAIConfig

    cfg = GourAIConfig(
        vocab_size=65,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.2,
        bias=False,
    )
    model = GourAILM(cfg)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("================ GourAI Master Model Assembly ================")
    print(f"Total Parameters         : {total_params:,} (~{total_params/1e6:.1f} Million)")
    print(f"Non-pos Parameters       : {model.get_num_params():,}")
    print(f"Layers (Blocks)          : {cfg.n_layer}")
    print(f"Heads per Block          : {cfg.n_head}")
    print(f"Embedding Dimensions    : {cfg.n_embd}")

    # Forward pass test with dummy inputs & targets
    dummy_idx = torch.randint(0, 65, (4, 16))
    dummy_targets = torch.randint(0, 65, (4, 16))

    logits, loss = model(dummy_idx, dummy_targets)
    print(f"Logits shape (B, T, V)   : {logits.shape} (Expected: [4, 16, 65])")
    print(f"Initial Cross-Entropy Loss: {loss.item():.4f}")
    
    # Starting loss for 65 classes should be around -ln(1/65) ≈ 4.17
    expected_loss = -math.log(1.0 / 65)
    print(f"Theoretical Starting Loss: {expected_loss:.4f}")
    assert abs(loss.item() - expected_loss) < 1.0, "Initial loss is abnormally far from theoretical random guessing!"
    print("[SUCCESS] GourAILM passed full forward pass & loss verification!")
    print("==============================================================")
