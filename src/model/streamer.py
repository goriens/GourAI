"""
Streaming text generator interface for GourAI V1.
Yields tokens incrementally as they are sampled, enabling real-time
typewriter response streaming for CLI, FastAPI SSE endpoints, and Web Playground.
"""
from typing import Generator, Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.model.generator import top_k_top_p_filtering


@torch.no_grad()
def stream_generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Generator[str, None, None]:
    """
    Yields decoded text tokens one-by-one in real-time.
    """
    model.eval()
    dev = device if device is not None else next(model.parameters()).device
    block_size = getattr(model.config, "block_size", 256)

    # Encode initial prompt
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        # Fallback if empty prompt: start with newline or space
        prompt_ids = [0]

    idx = torch.tensor(prompt_ids, dtype=torch.long, device=dev).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop context to model's context window limit
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Forward pass for next token logits
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Repetition penalty
        if repetition_penalty != 1.0:
            for prev_token in set(idx[0].tolist()):
                if logits[0, prev_token] < 0:
                    logits[0, prev_token] *= repetition_penalty
                else:
                    logits[0, prev_token] /= repetition_penalty

        # Temperature & Sampling
        if temperature > 0.0:
            logits = logits / temperature
            k_val = top_k if top_k is not None else 0
            p_val = top_p if top_p is not None else 1.0
            if k_val > 0 or p_val < 1.0:
                logits = top_k_top_p_filtering(logits, top_k=k_val, top_p=p_val)

            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token_id = torch.argmax(logits, dim=-1).item()

        # Append to running tensor sequence
        next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=dev)
        idx = torch.cat((idx, next_tensor), dim=1)

        # Decode single token chunk and yield
        token_str = tokenizer.decode([next_token_id])
        yield token_str


# Standalone self-verification test
if __name__ == "__main__":
    from src.config import GourAIConfig
    from src.model.transformer import GourAILM
    from src.tokenizer import get_tokenizer

    print("================ GourAI Streamer Verification ================")
    config = GourAIConfig(
        vocab_size=65,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model = GourAILM(config)
    tokenizer = get_tokenizer()

    prompt = "KING:"
    print(f"Prompt               : '{prompt}'")
    print("Streaming Output     : ", end="", flush=True)

    streamed_chars = []
    for char in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=15,
        temperature=0.8,
        top_k=20,
    ):
        print(char, end="", flush=True)
        streamed_chars.append(char)
    print("\n" + "-" * 62)

    assert len(streamed_chars) == 15, f"Expected 15 streamed characters, got {len(streamed_chars)}"
    print("[SUCCESS] GourAI Streamer passed all self-verification checks!")
    print("==============================================================")
