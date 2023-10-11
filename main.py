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
from src.utils import set_seeds, get_device, save_checkpoint

device = get_device()
set_seeds(seed_no=42)

print(device)

with open("sweep_params.yaml") as file:
    sweep_params = yaml.load(file, Loader=yaml.FullLoader)


run = wandb.init(project="mha-based-fsgnn-v2", config=sweep_params)
global c
c = wandb.config

ckpt = dict(params=c)
torch.save(ckpt, "ckpt_trial.pth")
# %%
ckpt = torch.load("ckpt_trial.pth")

# %%

dataset_name = c.dataset["dataset_name"]
data = GraphDataset(dataset_name=dataset_name)

MASK_MATRIX_CACHE_DIR = (
    Path.cwd() / "mask_matrix_cache" / f"{dataset_name}_mask_matrix_list.pth"
)

mask_matrix_list_full = torch.load(MASK_MATRIX_CACHE_DIR)
L = c.dataset["max_hop"] * 2 + 1
if dataset_name == "Citeseer" or dataset_name == "Cora":
    mask_matrix_list = mask_matrix_list_full[:L:2]
    #! if we use citeseer, we skip power of A_sym since it is problematic, we only use A_sym_tilde powers
    L = len(mask_matrix_list)

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
    n_heads=L,
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
    early_stop_verbose=True,
)

ckpt = dict(trainer=trainer, params=c)
save_checkpoint(ckpt, dataset_name, trainer.best_test_acc)
