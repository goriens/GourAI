"""
GourAI V1 Text Generation CLI.
Loads trained model weights from checkpoint and generates text autoregressively
with sampling controls (temperature, top-k, top-p) and real-time streaming output.
"""
import sys
import time
import argparse
from pathlib import Path
import torch

from src.config import GourAIConfig
from src.device import get_device, set_seed
from src.model.transformer import GourAILM
from src.model.streamer import stream_generate
from src.model.generator import generate
from src.tokenizer import get_tokenizer
from src.train.checkpoint import load_checkpoint


def run_generation(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device(args.device)

    # 1. Load Tokenizer
    tokenizer = get_tokenizer()

    # 2. Locate and load model checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        for candidate in [Path("checkpoints/model.pt"), Path("checkpoints/latest_ckpt.pt")]:
            if candidate.exists():
                ckpt_path = candidate
                print(f"[INFO] Using fallback '{candidate}'.")
                break
        else:
            print(f"[WARNING] No checkpoint found. Running with untrained initialized weights.")
            ckpt_path = None

    if ckpt_path is not None:
        checkpoint = load_checkpoint(str(ckpt_path), device=device)
        if checkpoint.get("config"):
            cfg = GourAIConfig.from_dict(checkpoint["config"])
        else:
            cfg = GourAIConfig()
        model = GourAILM(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[LOADED] Model weights restored from '{ckpt_path}'")
        if "best_val_loss" in checkpoint:
            print(f"[METRIC] Checkpoint Val Loss: {checkpoint['best_val_loss']:.4f}")
    else:
        cfg = GourAIConfig()
        model = GourAILM(cfg)

    model.to(device)
    model.eval()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print(">> GourAI V1 Text Generation")
    print("=" * 65)
    print(f"Prompt        : \"{args.prompt}\"")
    print(f"Sampling Args : Temp={args.temp} | Top-K={args.top_k} | Top-P={args.top_p} | Max Tokens={args.max_tokens}")
    print(f"Device        : {device}")
    print("-" * 65)
    print(args.prompt, end="", flush=True)

    t0 = time.time()
    num_tokens = 0

    if args.stream:
        # Real-time streaming generation
        for token_chunk in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        ):
            print(token_chunk, end="", flush=True)
            num_tokens += 1
    else:
        # Batch generation
        prompt_ids = torch.tensor(
            tokenizer.encode(args.prompt), dtype=torch.long, device=device
        ).unsqueeze(0)
        output_ids = generate(
            model=model,
            idx=prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        gen_tokens = output_ids[0, prompt_ids.size(1):].tolist()
        num_tokens = len(gen_tokens)
        print(tokenizer.decode(gen_tokens), end="", flush=True)

    dt = time.time() - t0
    tok_per_sec = num_tokens / dt if dt > 0 else 0

    print("\n" + "-" * 65)
    print(f"[STATS] Generated {num_tokens} tokens in {dt:.2f}s ({tok_per_sec:.1f} tokens/sec)")
    print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="GourAI V1 Text Generation CLI")
    parser.add_argument("--prompt", type=str, default="ROMEO:\n", help="Seed prompt string")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to .pt checkpoint")
    parser.add_argument("--max_tokens", type=int, default=150, help="Maximum new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (0.0 = greedy, 1.0 = standard)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K token truncation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (nucleus) cumulative threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (1.0 = none)")
    parser.add_argument("--stream", action="store_true", default=True, help="Stream tokens to stdout")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=1337, help="Random reproducibility seed")
    return parser.parse_args()


if __name__ == "__main__":
    run_generation(parse_args())
