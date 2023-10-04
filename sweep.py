# %%
# from dotenv import load_dotenv

# load_dotenv()

import torch
import wandb
import random
import numpy as np


from src.trainer import Trainer
from src.configurations.model_configs import MHAConfig, MLPConfig
from src.dataset import GraphDataset
from src.dataset import get_mask_matrix_list
from src.models import MHAbasedFSGNN


import yaml

with open("sweep_params.yaml") as file:
    sweep_params = yaml.load(file, Loader=yaml.FullLoader)


# %%
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seeds(42)


# Calculate mask matrix list combinations up to 8th power


def calculate_all_combinations(dataset_name="Cora", max_of_max_hop: int = 8):
    data = GraphDataset(dataset_name=dataset_name)
    all_combinations = {}
    for i in range(max_of_max_hop + 1):
        all_combinations[f"up to {i}"] = get_mask_matrix_list(
            data.A_sym, data.A_sym_tilde, max_hop=i
        )


all_c = calculate_all_combinations(dataset_name="Cora", max_of_max_hop=8)


def run_sweep(c: dict = None):
    global sweep_id

    run = wandb.init(config=c)
    c = wandb.config

    data = GraphDataset(dataset_name=c.dataset["dataset_name"])

    # mask_matrix_list = get_mask_matrix_list(
    #     data.A_sym, data.A_sym_tilde, max_hop=c.dataset["max_hop"]
    # )

    mask_matrix_list = all_c[f"up to {c.dataset['max_hop']}"]

    mlp_config = MLPConfig(
        in_dim=data.n_feats,
        hidden_dims=c.mlp["hidden_dims"],
        out_dim=c.mlp["out_dim"],
        dropout=c.mlp["dropout"],
    )

    mha_config = MHAConfig(
        fan_in=c.mlp["out_dim"],
        fan_out=c.mha["fan_out"],
        n_heads=c.dataset["max_hop"] * 2 + 1,
        p=c.mha["p"],
        mask_matrix_list=mask_matrix_list,
    )

    model = MHAbasedFSGNN(
        mlp_config=mlp_config,
        mha_config=mha_config,
        n_class=data.n_class,
        skip_connection=c.skip_connection,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=c.optimizer["lr"],
        weight_decay=c.optimizer["weight_decay"],
    )

    trainer = Trainer(model=model, optimizer=optimizer, data=data)

    trainer.pipeline(
        max_epochs=c.trainer_pipeline["max_epochs"],
        patience=c.trainer_pipeline["patience"],
        wandb_flag=True,
        sweep_id=sweep_id,
        early_stop_verbose=True,
    )


sweep_id = wandb.sweep(sweep_params, project="mha-based-fsgnn")
wandb.agent(sweep_id, function=run_sweep)
