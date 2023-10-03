# %%
import torch
from src.trainer import Trainer
from src.dataset import GraphDataset, get_mask_matrix_list
from src.models import MHAbasedFSGNN
from src.configurations.model_configs import MLPConfig, MHAConfig
import random
import numpy as np


def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed) 
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


set_seeds(42)

#! This file used for debugging purposes only, to run sweep experiments use sweep.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

print(device)
data = GraphDataset(dataset_name="Cora")
data.print_dataset_info()

# %%

max_hop = 2
L = 2 * max_hop + 1

mask_matrix_list = get_mask_matrix_list(data.A_sym, data.A_sym_tilde, max_hop=max_hop)
mask_matrix_list = [m.to(device) for m in mask_matrix_list]

mlp_config = MLPConfig(
    in_dim=data.n_feats,
    hidden_dims=500,
    out_dim=64,
    dropout=0.5,
    normalization=torch.nn.LayerNorm,
)

mha_config = MHAConfig(
    fan_in=64, fan_out=64, n_heads=L, p=0.4, mask_matrix_list=mask_matrix_list
)

model = MHAbasedFSGNN(
    mlp_config=mlp_config,
    mha_config=mha_config,
    n_class=data.n_class,
    skip_connection=True,

)
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

trainer = Trainer(model=model, optimizer=optimizer, data=data)

trainer.pipeline(
    max_epochs=500, patience=100, wandb_flag=False, early_stop_verbose=True
)

# %%
