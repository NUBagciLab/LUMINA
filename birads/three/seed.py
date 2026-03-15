# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:25:58 2024

@author: pky0507
"""

import random
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)                          # Set seed for Python's random
    np.random.seed(seed)                       # Set seed for NumPy
    torch.manual_seed(seed)                    # Set seed for CPU
    torch.cuda.manual_seed(seed)               # Set seed for current GPU
    torch.cuda.manual_seed_all(seed)           # Set seed for all GPUs (if you are using more than one)
    torch.backends.cudnn.deterministic = True  # Enable deterministic mode
    torch.backends.cudnn.benchmark = False     # Disable the benchmark for reproducibility