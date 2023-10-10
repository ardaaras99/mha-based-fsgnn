import random
import torch
import numpy as np


def set_seeds(seed_no: int = 42):
    # torch.backends.cudnn.deterministic = True
    random.seed(seed_no)
    np.random.seed(seed_no)
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed_all(seed_no)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
