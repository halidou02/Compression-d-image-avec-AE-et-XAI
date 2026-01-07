"""
Seed utilities for reproducibility.
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get available device (CUDA if available, else CPU).
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    set_seed(42)
    print(f"Device: {get_device()}")
    print(f"Random: {random.random():.4f}")
    print(f"Numpy: {np.random.rand():.4f}")
    print(f"Torch: {torch.rand(1).item():.4f}")
