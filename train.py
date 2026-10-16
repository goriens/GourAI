"""
Master Training Script for GourAI V1.
Assembles the complete training loop:
- Memory-mapped batch slicing (B x T)
- Cosine decay learning rate scheduler with linear warmup
- AdamW optimizer with separated weight decay
- Gradient clipping (1.0)
- Validation loss evaluation & atomic checkpointing (checkpoints/best_model.pt)
"""
import os
import time
import math
import argparse
from pathlib import Path
import torch
import torch.nn as nn

from src.config import GourAIConfig
from src.device import get_device, set_seed, get_device_info
from src.model.transformer import GourAILM
from src.train.dataloader import GourAIDataLoader
from src.train.optimizer import configure_optimizers
from src.train.scheduler import get_lr, update_lr
from src.train.checkpoint import save_checkpoint, restore_model_from_checkpoint


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    loader: GourAIDataLoader,
    eval_iters: int,
    batch_size: int,
    block_size: int,
) -> dict:
    """
    Evaluates model across train and val splits over eval_iters random batches.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = loader.get_batch(split, batch_size=batch_size, block_size=block_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(args: argparse.Namespace):
    # 1. Enforce reproducibility seed
    set_seed(args.seed)

    # 2. Determine computation device
    device = get_device(args.device)
    device_type = device.type
    print("================ GourAI Training Engine ================")
    print(f"Device               : {device} ({get_device_info().get('gpu_name', 'CPU')})")
    print(f"Random Seed          : {args.seed}")

    # 3. Initialize configuration
    cfg = GourAIConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters if args.lr_decay_iters else args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
    )

    # 4. Initialize DataLoader
    loader = GourAIDataLoader(data_dir=cfg.data_dir, device=device)
    print(f"Dataset Loaded       : {len(loader.train_data):,} train tokens | {len(loader.val_data):,} val tokens")

    # 5. Initialize Model
    model = GourAILM(cfg)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters     : {total_params:,} (~{total_params/1e6:.2f}M)")

    # 6. Configure Optimizer
    optimizer, (num_decay, num_nodecay, use_fused) = configure_optimizers(
        model,
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        device_type=device_type,
    )
    print(f"Optimizer Groups     : {num_decay:,} decay / {num_nodecay:,} no-decay (Fused: {use_fused})")

    iter_num = 0
    best_val_loss = float("inf")
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / cfg.best_model_name
    latest_ckpt_path = checkpoint_dir / "latest_ckpt.pt"

    # 7. Optionally resume from checkpoint
    if args.resume and os.path.exists(latest_ckpt_path):
        iter_num, best_val_loss, _ = restore_model_from_checkpoint(
            str(latest_ckpt_path), model, optimizer, device=device
        )
        print(f"[RESUME] Continuing training from iteration {iter_num}...")

    # 8. Training loop
    print("\nStarting Training Loop...")
    print("-" * 75)
    t0 = time.time()

    while iter_num < cfg.max_iters:
        # Determine and set learning rate for this step
        lr = get_lr(
            iter_num,
            learning_rate=cfg.learning_rate,
            min_lr=cfg.min_lr,
            warmup_iters=cfg.warmup_iters,
            lr_decay_iters=cfg.lr_decay_iters,
        )
        update_lr(optimizer, lr)

        # Periodic evaluation & checkpointing
        if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters - 1:
            losses = estimate_loss(
                model, loader, cfg.eval_iters, cfg.batch_size, cfg.block_size
            )
            print(
                f"Step {iter_num:5d}/{cfg.max_iters} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"LR: {lr:.2e}"
            )

            # Save best model if validation loss improved
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    str(best_ckpt_path),
                    model=model,
                    optimizer=optimizer,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss,
                    config=cfg,
                )

            # Save latest checkpoint for resume support
            save_checkpoint(
                str(latest_ckpt_path),
                model=model,
                optimizer=optimizer,
                iter_num=iter_num,
                best_val_loss=best_val_loss,
                config=cfg,
            )

        # Fetch training mini-batch
        x, y = loader.get_batch(
            split="train", batch_size=cfg.batch_size, block_size=cfg.block_size
        )

        # Forward pass
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping to avoid exploding gradients
        if cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # Optimizer step
        optimizer.step()

        # Step timing & progress log
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % 10 == 0 and iter_num > 0:
            tokens_per_sec = (cfg.batch_size * cfg.block_size) / dt
            print(
                f"Iter {iter_num:5d} | "
                f"Loss: {loss.item():.4f} | "
                f"Step Time: {dt*1000:6.1f}ms | "
                f"Speed: {tokens_per_sec:6.0f} tok/s"
            )

        iter_num += 1

    print("-" * 75)
    print(f"[DONE] Training finished! Best Validation Loss: {best_val_loss:.4f}")
    print(f"[SAVED] Production weights available at '{best_ckpt_path}'")
    print("========================================================")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GourAI V1 Transformer Model")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--block_size", type=int, default=256, help="Context window length")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of Transformer blocks")
    parser.add_argument("--n_head", type=int, default=6, help="Number of Attention heads")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--vocab_size", type=int, default=65, help="Vocabulary size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--bias", action="store_true", help="Enable bias in Linear layers")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Max learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Min learning rate")
    parser.add_argument("--max_iters", type=int, default=1000, help="Total training steps")
    parser.add_argument("--warmup_iters", type=int, default=50, help="Warmup steps")
    parser.add_argument("--lr_decay_iters", type=int, default=1000, help="LR decay steps")
    parser.add_argument("--eval_interval", type=int, default=100, help="Eval interval")
    parser.add_argument("--eval_iters", type=int, default=10, help="Eval batches count")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
