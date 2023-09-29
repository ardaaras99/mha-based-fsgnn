import torch.nn as nn
from torch.nn import functional as F
import torch


class Head(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_middle: int,
        fan_out: int,
    ):
        super(Head, self).__init__()

        # adding pre_proj layer to reduce the dimension of input

        self.W_proj = nn.Linear(fan_in, fan_middle, bias=False)
        self.W_K = nn.Linear(fan_middle, fan_out, bias=False)
        self.W_Q = nn.Linear(fan_middle, fan_out, bias=False)
        self.W_V = nn.Linear(fan_middle, fan_out, bias=False)

        self._set_parameters()

    def forward(self, X):
        X = self.W_proj(X)
        K = self.W_K(X)
        Q = self.W_Q(X)
        V = self.W_V(X)
        E = (Q @ K.T) / (K.shape[-1] ** 0.5)
        alpha_mat = F.softmax(E, dim=-1)
        H = alpha_mat @ V
        return H

    def _set_parameters(self):
        nn.init.kaiming_uniform_(self.W_K.weight)
        nn.init.kaiming_uniform_(self.W_Q.weight)
        nn.init.kaiming_uniform_(self.W_V.weight)
        nn.init.kaiming_uniform_(self.W_proj.weight)


class MHA(nn.Module):
    def __init__(self, fan_in: int, fan_middle: int, n_heads: int, p: float):
        super(MHA, self).__init__()
        self.n_heads = n_heads
        head_dim = fan_middle // n_heads
        self.p = p

        self.heads = nn.ModuleList(
            [
                Head(fan_in=fan_in, fan_middle=fan_middle, fan_out=head_dim)
                for _ in range(n_heads)
            ]
        )
        # for aggregation of heads
        self.W_O = nn.Linear(fan_middle, fan_middle, bias=False)

    def forward(self, input_list):
        out = torch.cat(
            [head(input_list[i]) for i, head in enumerate(self.heads)], dim=-1
        )
        out = self.W_O(out)
        out = F.dropout(out, p=self.p, training=self.training)
        return out


class Model(nn.Module):
    def __init__(self, fan_in: int, fan_middle: int, n_heads: int, n_class: int):
        super(Model, self).__init__()
        self.n_heads = n_heads

        self.mha = MHA(fan_in=fan_in, fan_middle=fan_middle, n_heads=n_heads, p=0.6)
        self.ln1 = nn.LayerNorm(fan_in)
        self.ln2 = nn.LayerNorm(fan_middle)

        self.proj_out = nn.Linear(fan_middle, n_class)

    def forward(self, input_list):
        input_list = [self.ln1(input_list[i]) for i in range(len(input_list))]
        out = self.mha(input_list)
        out = self.ln2(out)

        out = F.elu(out)
        out = F.dropout(out, p=0.6, training=self.training)
        out = self.proj_out(out)
        out = F.log_softmax(out, dim=1)
        return out
