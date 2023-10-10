# %%
import torch
import random
import numpy as np
import yaml

from src.trainer import Trainer
from src.dataset import GraphDataset, get_mask_matrix_list
from src.models import MHAbasedFSGNN
from src.configurations.model_configs import MLPConfig, MHAConfig
from src.utils import set_seeds, get_device

#! This file used for debugging purposes only, to run sweep experiments use sweep.py


with open("sweep_params.yaml") as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)

set_seeds(42)
device = get_device()
print(device)

# Extract configs
mlp_config = sweep_config["mlp"]
mha_config = sweep_config["mha"]
dataset_config = sweep_config["dataset"]


data = GraphDataset(dataset_name=dataset_config["dataset_name"])
data.print_dataset_info()


# %%

max_hop = dataset_config["max_hop"]
L = 2 * max_hop + 1

mask_matrix_list = get_mask_matrix_list(data.A_sym, data.A_sym_tilde, max_hop=max_hop)
mask_matrix_list = [m.to(device) for m in mask_matrix_list]

mlp_config = MLPConfig(
    in_dim=data.n_feats,
    hidden_dims=mlp_config["hidden_dims"],
    out_dim=mlp_config["out_dim"],
    dropout=mlp_config["dropout"],
)

mha_config = MHAConfig(
    fan_in=mlp_config["out_dim"],
    fan_out=mha_config["fan_out"],
    n_heads=L,
    p=mha_config["p"],
    mask_matrix_list=mask_matrix_list,
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
