"""
Device and hardware acceleration configuration for GourAI V1.
Supports automatic selection of CUDA, MPS (Apple Silicon), or CPU fallback.
"""
import torch
import random
import os


def get_device(requested_device: str = "auto") -> torch.device:
    """
    Returns torch.device based on requested preference or automated hardware detection.
    """
    if requested_device != "auto":
        return torch.device(requested_device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        if hasattr(os, "cpu_count") and os.cpu_count():
            try:
                torch.set_num_threads(os.cpu_count())
            except Exception:
                pass
        return torch.device("cpu")


def set_seed(seed: int = 1337) -> None:
    """
    Enforces deterministic reproducibility across PyTorch and Python standard library.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_info() -> dict:
    """
    Returns diagnostic information about current hardware and PyTorch runtime.
    """
    device = get_device()
    info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


if __name__ == "__main__":
    set_seed(1337)
    info = get_device_info()
    print("================ GourAI Hardware Diagnostic ================")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("============================================================")
