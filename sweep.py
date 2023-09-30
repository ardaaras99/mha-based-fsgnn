# %%
# from dotenv import load_dotenv

# load_dotenv()

import torch
import wandb
import yaml
import random
import numpy as np


from src.engine.trainer import Trainer
from src.configurations.model_configs import MHAConfig, MLPConfig
from src.dataset import GraphDataset
from src.utils.input_generation import get_mask_matrix_list
from src.models.models import MHAbasedFSGNN


with open("src/configurations/sweep_config.yaml") as file:
    sweep_config = yaml.load(file, Loader=yaml.FullLoader)


# %%


def set_seeds():
    # torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


set_seeds()


def run_sweep(config: dict = None):
    global sweep_id

    run = wandb.init(config=config)
    config = wandb.config

    data = GraphDataset(dataset_name=config.dataset_name)
    data.print_dataset_info()

    max_hop = 2
    L = 2 * max_hop + 1

    mask_matrix_list = get_mask_matrix_list(
        data.A_sym, data.A_sym_tilde, max_hop=max_hop
    )

    #! Since we use skip connection, chose L wisely so that head concatenation will not reduce dimension
    # for max_hop = 3, L = 7, if we choose fan_in = 64, we loose dimension

    mlp_config = MLPConfig(
        in_dim=data.n_feats,
        hidden_dims=300,
        out_dim=64,
        dropout=0.5,
        normalization=torch.nn.LayerNorm,
    )

    mha_config = MHAConfig(
        fan_in=64, fan_out=64, n_heads=L, p=0.4, mask_matrix_list=mask_matrix_list
    )

    model = MHAbasedFSGNN(
        mlp_config=mlp_config, mha_config=mha_config, n_class=data.n_class
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    from engine.trainer import Trainer

    trainer = Trainer(model=model, optimizer=optimizer, data=data)

    trainer.pipeline(
        max_epochs=500, patience=100, wandb_flag=False, early_stop_verbose=True
    )


sweep_id = wandb.sweep(sweep_config, project="mha-based-fsgnn")
wandb.agent(sweep_id, function=run_sweep, count=4)
