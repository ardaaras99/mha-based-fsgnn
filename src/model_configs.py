import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Optional, Union


@dataclass
class MHAConfig:
    fan_in: int
    fan_out: int
    n_heads: int
    p: float
    mask_matrix_list: list[torch.Tensor]


@dataclass
class MLPConfig:
    in_dim: int
    out_dim: int
    hidden_dims: Optional[Union[int, list[int]]] = None
    dropout: float = 0.5
    activation: Callable[..., nn.Module] = nn.ReLU
    normalization: Optional[Callable[..., nn.Module]] = nn.LayerNorm
