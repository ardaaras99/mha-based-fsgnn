# %%
# from dotenv import load_dotenv
# load_dotenv()

import torch
import wandb
import yaml
from pathlib import Path


from src.trainer import Trainer
from src.configurations.model_configs import MHAConfig, MLPConfig
from src.dataset import GraphDataset
from src.models import MHAbasedFSGNN
from src.utils import set_seeds, get_device


device = get_device()
set_seeds(seed_no=42)

print(device)

with open("sweep_params.yaml") as file:
    sweep_params = yaml.load(file, Loader=yaml.FullLoader)


def run_sweep(c: dict = None):
    global sweep_id

    run = wandb.init(config=c)
    c = wandb.config

    dataset_name = c.dataset["dataset_name"]
    data = GraphDataset(dataset_name=dataset_name)

    MASK_MATRIX_CACHE_DIR = (
        Path.cwd() / "mask_matrix_cache" / f"{dataset_name}_mask_matrix_list.pth"
    )

    mask_matrix_list_full = torch.load(MASK_MATRIX_CACHE_DIR)
    mask_matrix_list = mask_matrix_list_full[: 2 * c.dataset["max_hop"] + 1]
    mask_matrix_list = [m.to(device) for m in mask_matrix_list]

    mlp_config = MLPConfig(
        in_dim=data.n_feats,
        hidden_dims=c.mlp["hidden_dims"],
        out_dim=c.mlp["out_dim"],
        dropout=c.mlp["dropout"],
    )

    fan_in = c.mlp["out_dim"]
    mha_config = MHAConfig(
        fan_in=fan_in,
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

    model.to(device)

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

# %%
