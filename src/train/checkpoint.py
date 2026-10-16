"""
Checkpoint management system for GourAI V1.
Handles atomic saving and restoring of model weights, optimizer states,
training progress metrics, and model architecture hyperparameters.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from src.config import GourAIConfig


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    iter_num: int = 0,
    best_val_loss: float = float("inf"),
    config: Optional[GourAIConfig] = None,
) -> None:
    """
    Saves complete training checkpoint containing weights, optimizer state, and metadata.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap raw model if wrapped (e.g. DistributedDataParallel or compile)
    raw_model = getattr(model, "module", model)
    raw_model = getattr(raw_model, "_orig_mod", raw_model)

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config.to_dict() if config is not None else None,
    }

    # Save to temporary path first for atomic write safety
    temp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)
    print(f"[CHECKPOINT] Saved checkpoint to '{filepath}' (Iter: {iter_num}, Val Loss: {best_val_loss:.4f})")


def load_checkpoint(
    filepath: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Loads checkpoint dictionary from disk.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return checkpoint


def restore_model_from_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float, Optional[GourAIConfig]]:
    """
    Restores model and optional optimizer from a checkpoint file.
    Returns (iter_num, best_val_loss, config).
    """
    checkpoint = load_checkpoint(filepath, device=device)
    
    raw_model = getattr(model, "module", model)
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    raw_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    
    config = None
    if checkpoint.get("config") is not None:
        config = GourAIConfig.from_dict(checkpoint["config"])

    print(f"[CHECKPOINT] Restored weights from '{filepath}' (Iter: {iter_num}, Loss: {best_val_loss:.4f})")
    return iter_num, best_val_loss, config


# Standalone self-verification test
if __name__ == "__main__":
    import tempfile
    import os
    from src.model.transformer import GourAILM

    print("================ GourAI Checkpoint Verification ================")
    config = GourAIConfig(
        vocab_size=65,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model1 = GourAILM(config)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_ckpt.pt")

        # Save checkpoint
        save_checkpoint(
            filepath=ckpt_path,
            model=model1,
            optimizer=optimizer1,
            iter_num=42,
            best_val_loss=2.3456,
            config=config,
        )
        assert os.path.exists(ckpt_path), "Checkpoint file was not created!"

        # Initialize fresh model2 and restore
        model2 = GourAILM(config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        # Confirm model1 and model2 have different initial weights
        p1 = next(model1.parameters()).data
        p2 = next(model2.parameters()).data
        assert not torch.equal(p1, p2), "Initial random models should not have identical weights"

        # Restore into model2
        iter_num, best_loss, restored_cfg = restore_model_from_checkpoint(
            ckpt_path, model2, optimizer2
        )

        assert iter_num == 42
        assert abs(best_loss - 2.3456) < 1e-4
        assert restored_cfg.n_embd == 64

        # Verify weights match after restoring
        p2_restored = next(model2.parameters()).data
        assert torch.equal(p1, p2_restored), "Model weights mismatch after checkpoint restore!"
        print("  State dict match verified across models.")

    print("[SUCCESS] GourAI Checkpoint manager passed all self-verification checks!")
    print("==================================================================")
