import torch.nn as nn
from torch.nn import functional as F
import torch


from src.model_configs import MLPConfig, MHAConfig


class Head(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        mask_matrix: torch.Tensor,
    ):
        super(Head, self).__init__()
        self.W_K = nn.Linear(fan_in, fan_out, bias=False)
        self.W_Q = nn.Linear(fan_in, fan_out, bias=False)
        self.W_V = nn.Linear(fan_in, fan_out, bias=False)
        self.mask_matrix = mask_matrix

        self._set_parameters()

    def forward(self, X):
        K, Q, V = self.W_K(X), self.W_Q(X), self.W_V(X)
        E = (Q @ K.T) / (K.shape[-1] ** 0.5)
        E = E.masked_fill(self.mask_matrix == 0, float("-inf"))
        #! Attention output can be treated as soft adjacency matrix
        A_soft = F.softmax(E, dim=-1)
        H = A_soft @ V
        return H

    def _set_parameters(self):
        nn.init.kaiming_uniform_(self.W_K.weight)
        nn.init.kaiming_uniform_(self.W_Q.weight)
        nn.init.kaiming_uniform_(self.W_V.weight)


class MHA(nn.Module):
    def __init__(
        self,
        config: MHAConfig,
    ):
        super(MHA, self).__init__()
        """
        fan_in: input dimension
        fan_out: output dimension
        n_heads: number of heads
        p: dropout probability
        mask_matrix_list: list of mask matrices for each head, depends on the input

        example: 
            input H: [N, fan_in], fan_in, fan_out, n_heads
            output: H_out = [N, fan_out]
            each head must be of head_dim: [N, fan_out // n_heads] to have
            
            note: to have skip connections make fan_in = fan_out
        """
        self.config = config
        head_dim = config.fan_out // config.n_heads

        l = nn.ModuleList()
        for i in range(config.n_heads):
            l.append(
                Head(
                    fan_in=config.fan_in,
                    fan_out=head_dim,
                    mask_matrix=config.mask_matrix_list[i],
                )
            )

        self.heads = nn.Sequential(*l)

        # for aggregation of heads
        ## input is H: [N, fan_out]
        ## output can be H_out: [N, arbitrary_dimension], this can be tuned, preserve as fan_out for now
        self.W_O = nn.Linear(head_dim * config.n_heads, config.fan_out, bias=False)

    def forward(self, input_list):
        out_list = []
        for i, head in enumerate(self.heads):
            out_list.append(head(input_list[i]))
        out = torch.cat(out_list, dim=-1)
        out = self.W_O(out)
        out = F.dropout(out, p=self.config.p, training=self.training)
        return out


class MLP(nn.Module):
    # for reference: https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
    def __init__(
        self,
        config: MLPConfig,
        **kwargs,
    ) -> None:
        super().__init__()

        l = nn.ModuleList()

        if config.hidden_dims is None:
            config.hidden_dims = []

        if isinstance(config.hidden_dims, int):
            config.hidden_dims = [config.hidden_dims]

        in_dim = config.in_dim
        for hidden_dim in config.hidden_dims:
            l.append(nn.Linear(in_dim, hidden_dim))
            if config.normalization:
                l.append(config.normalization(hidden_dim))
            l.append(config.activation())
            l.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        l.append(nn.Linear(in_dim, config.out_dim))
        #! Last dropout layer added by Arda Can Aras
        # l.append(nn.Dropout(config.dropout))

        self.model = nn.Sequential(*l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MHAbasedFSGNN(nn.Module):
    def __init__(self, mlp_config: MLPConfig, mha_config: MHAConfig, n_class: int):
        super(MHAbasedFSGNN, self).__init__()

        self.mha_config = mha_config

        self.MLP = MLP(config=mlp_config)
        self.mha = MHA(config=mha_config)

        self.ln = nn.LayerNorm(mha_config.fan_out)
        self.clf_head = nn.Linear(mha_config.fan_out, n_class)

    def forward(self, X):
        X = self.MLP(X)
        input_list = []
        for A in self.mha_config.mask_matrix_list:
            input_list.append(A @ X)

        out = self.mha(input_list)
        #! skip connection
        # assert X.shape == out.shape
        # out = X + out

        #! Retrieved from GAT architecture might not be necessary
        # Layer Normalization
        # out = self.ln(out)
        # L2 Normalization
        out = F.normalize(out, p=2, dim=-1)
        out = F.relu(out)
        # out = F.dropout(out, p=0.6, training=self.training)
        out = self.clf_head(out)
        out = F.log_softmax(out, dim=1)
        return out
