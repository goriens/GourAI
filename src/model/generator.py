"""
Autoregressive Generation Engine for GourAI V1.
Implements next-token autoregressive sampling with:
- Context window cropping (block_size limit)
- Temperature scaling (variance control)
- Top-K filtering (sampling space truncation)
- Top-P / Nucleus sampling (cumulative probability cutoff)
- Repetition penalty support
"""
from typing import Optional, List
import torch
import torch.nn as nn
from torch.nn import functional as F


@torch.no_grad()
def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """
    Filters logits using Top-K and/or Nucleus (Top-P) sampling algorithms.
    """
    logits = logits.clone()

    # Top-K filtering: keep only top k tokens
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-P (Nucleus) filtering: keep cumulative probability <= top_p tokens
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to the right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Generates new tokens autoregressively from conditioned token sequence idx (B, T).
    """
    model.eval()
    block_size = getattr(model.config, "block_size", 256)

    for _ in range(max_new_tokens):
        # Crop context to block_size if sequence exceeds maximum context window
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Forward pass (targets=None triggers fast (B, 1, V) logits path in GourAILM)
        logits, _ = model(idx_cond)
        # Isolate last token logits: (B, vocab_size)
        logits = logits[:, -1, :]

        # Apply repetition penalty if enabled (> 1.0)
        if repetition_penalty != 1.0:
            for b in range(idx.shape[0]):
                for prev_token in set(idx[b].tolist()):
                    if logits[b, prev_token] < 0:
                        logits[b, prev_token] *= repetition_penalty
                    else:
                        logits[b, prev_token] /= repetition_penalty

        # Temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Filter via Top-K / Top-P
            k_val = top_k if top_k is not None else 0
            p_val = top_p if top_p is not None else 1.0
            if k_val > 0 or p_val < 1.0:
                logits = top_k_top_p_filtering(logits, top_k=k_val, top_p=p_val)

            probs = F.softmax(logits, dim=-1)
            # Sample from categorical distribution
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding (Argmax) when temperature is 0
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append generated token to running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# Standalone self-verification test
if __name__ == "__main__":
    from src.config import GourAIConfig
    from src.model.transformer import GourAILM

    print("================ GourAI Generator Verification ================")
    config = GourAIConfig(
        vocab_size=65,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model = GourAILM(config)

    # Initial prompt tensor (Batch 1, Length 5)
    prompt_ids = torch.randint(0, 65, (1, 5))
    print(f"Prompt IDs shape           : {prompt_ids.shape}")

    # Generate 10 new tokens with temp=0.8, top_k=20, top_p=0.9
    gen_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
    )
    print(f"Generated IDs shape        : {gen_ids.shape} (Expected: [1, 15])")
    assert gen_ids.shape == (1, 15), "Output length mismatch in autoregressive generation!"

    # Test greedy decoding (temp=0.0)
    greedy_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=5,
        temperature=0.0,
    )
    assert greedy_ids.shape == (1, 10)
    print("  Greedy decoding (temperature=0.0) passed.")

    # Test context cropping (> block_size)
    long_prompt = torch.randint(0, 65, (1, 70))  # 70 > block_size 64
    cropped_out = generate(
        model,
        long_prompt,
        max_new_tokens=2,
        temperature=1.0,
    )
    assert cropped_out.shape == (1, 72)
    print("  Context window auto-cropping (> block_size) passed.")

    print("[SUCCESS] GourAI Generator passed all self-verification checks!")
    print("===============================================================")
