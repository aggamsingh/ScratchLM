"""Reproducibility: seed every RNG we use."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (CPU + all CUDA devices)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op without CUDA
