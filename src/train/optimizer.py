"""
Optimizer configuration for GourAI V1.
Sets up AdamW with weight decay separation:
- All 2D weight matrices (Linear projections, Embeddings) receive L2 weight decay.
- All 1D tensors (LayerNorm scales, biases) are exempt from weight decay (decay = 0.0).
Supports fused AdamW kernel when available on CUDA devices.
"""
import inspect
from typing import Tuple
import torch
import torch.nn as nn
from src.config import GourAIConfig


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 3e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.Optimizer:
    """
    Constructs and configures an AdamW optimizer instance with weight decay separation.
    """
    # Collect all parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Separate parameters into 2D (decay) and <2D (no decay)
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    # Sanity checks
    assert len(decay_params) + len(nodecay_params) == len(param_dict), (
        f"Mismatch in parameter count! Found {len(param_dict)} trainable parameters, "
        f"partitioned into {len(decay_params)} decay and {len(nodecay_params)} non-decay."
    )

    # Check whether fused AdamW is supported (CUDA + PyTorch >= 2.0)
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )

    return optimizer, (num_decay_params, num_nodecay_params, use_fused)


# Standalone self-verification test
if __name__ == "__main__":
    from src.model.transformer import GourAILM

    print("================ GourAI Optimizer Verification ================")
    config = GourAIConfig(
        vocab_size=65,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )
    test_model = GourAILM(config)

    optimizer, (num_decay, num_nodecay, use_fused) = configure_optimizers(
        test_model,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type="cpu",
    )

    print(f"Decayed parameters count     : {num_decay:,}")
    print(f"Non-decayed parameters count : {num_nodecay:,}")
    print(f"Fused kernel enabled         : {use_fused}")

    # Check parameter groups
    assert len(optimizer.param_groups) == 2, "Expected exactly 2 parameter groups"
    assert optimizer.param_groups[0]["weight_decay"] == config.weight_decay
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    print("  Weight decay groups verified (group 0: 0.1, group 1: 0.0)")

    # Execute dummy forward & backward step
    dummy_x = torch.randint(0, 65, (2, 16))
    dummy_y = torch.randint(0, 65, (2, 16))

    _, loss = test_model(dummy_x, dummy_y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"Dummy backward pass & step completed. Loss: {loss.item():.4f}")
    print("[SUCCESS] GourAI Optimizer configuration passed all checks!")
    print("================================================================")
