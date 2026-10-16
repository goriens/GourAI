"""
Cosine Decay with Linear Warmup Learning Rate Scheduler for GourAI V1.
Implements the GPT-2 / nanoGPT schedule:
1. Linear warmup from 0 to max_lr over warmup_iters.
2. Cosine decay from max_lr down to min_lr over (lr_decay_iters - warmup_iters).
3. Constant min_lr for iterations beyond lr_decay_iters.
"""
import math
import torch
from src.config import GourAIConfig


def get_lr(
    it: int,
    learning_rate: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_iters: int = 100,
    lr_decay_iters: int = 2000,
) -> float:
    """
    Computes learning rate for current step `it` using linear warmup and cosine decay.
    """
    # 1. Linear warmup phase
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    # 2. Beyond decay target, lock at minimum learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3. Cosine decay phase
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def update_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """
    Updates the learning rate for all parameter groups in the optimizer.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Standalone self-verification test
if __name__ == "__main__":
    print("================ GourAI LR Scheduler Verification ================")
    cfg = GourAIConfig(
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup_iters=100,
        lr_decay_iters=2000,
    )

    # Step 0 (Start of warmup)
    lr_start = get_lr(0, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.lr_decay_iters)
    print(f"Step 0 (Warmup start)     : {lr_start:.7f}")
    assert lr_start < cfg.learning_rate

    # Step warmup_iters (Peak LR)
    lr_peak = get_lr(cfg.warmup_iters, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.lr_decay_iters)
    print(f"Step {cfg.warmup_iters} (Peak LR)         : {lr_peak:.7f}")
    assert math.isclose(lr_peak, cfg.learning_rate, rel_tol=1e-3)

    # Step 1050 (Midway decay)
    lr_mid = get_lr(1050, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.lr_decay_iters)
    print(f"Step 1050 (Midway decay)  : {lr_mid:.7f}")
    assert cfg.min_lr < lr_mid < cfg.learning_rate

    # Step 2000 (End of decay)
    lr_end = get_lr(cfg.lr_decay_iters, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.lr_decay_iters)
    print(f"Step {cfg.lr_decay_iters} (End of decay)   : {lr_end:.7f}")
    assert math.isclose(lr_end, cfg.min_lr, rel_tol=1e-3)

    # Step 2500 (Post decay lock)
    lr_after = get_lr(2500, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.lr_decay_iters)
    print(f"Step 2500 (Beyond lock)   : {lr_after:.7f}")
    assert lr_after == cfg.min_lr

    print("[SUCCESS] GourAI LR Scheduler passed all self-verification checks!")
    print("==================================================================")
