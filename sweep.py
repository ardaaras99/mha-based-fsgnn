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
import os
import yaml


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

print(device)

with open("sweep_params.yaml") as file:
    sweep_params = yaml.load(file, Loader=yaml.FullLoader)


# %%
def set_seeds():
    # torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


set_seeds()





def run_sweep(c: dict = None):
    global sweep_id

    run = wandb.init(config=c)
    c = wandb.config

    data = GraphDataset(dataset_name=c.dataset["dataset_name"])

    A_sym = data.A_sym
    A_sym_tilde = data.A_sym_tilde

    mask_matrix_cache_file = 'mask_matrix_cache.npy'


    if os.path.exists(mask_matrix_cache_file):
        print('Loading pre-computed mask matrix cache') 
        MASK_MATRIX_CACHE = np.load(mask_matrix_cache_file).item()

    else:
        print('Gening mask mat')

        MASK_MATRIX_CACHE = {}
        
        max_hops = [2,3,4,5,6,7,8]

        for max_hop in max_hops:
        
            mask_matrix_list = get_mask_matrix_list(A_sym, A_sym_tilde, max_hop=max_hop)

            MASK_MATRIX_CACHE[max_hop] = mask_matrix_list

        np.save('mask_matrix_cache.npy', MASK_MATRIX_CACHE)
        print('Cache saved to mask_matrix_cache.npy')

    


    max_hop = c.dataset["max_hop"]
    mask_matrix_list = MASK_MATRIX_CACHE[max_hop]    

    #mask_matrix_list = get_mask_matrix_list(
        #data.A_sym, data.A_sym_tilde, max_hop=c.dataset["max_hop"])

    mask_matrix_list = [m.to(device) for m in mask_matrix_list]

    mlp_config = MLPConfig(
        in_dim=data.n_feats,
        hidden_dims=c.mlp["hidden_dims"],
        out_dim=c.mlp["out_dim"],
        dropout=c.mlp["dropout"],
    )

    mha_config = MHAConfig(
        fan_in=c.mha["fan_in"],
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
