import torch
import wandb
import yaml


from src.trainer import Trainer
from src.models import MHAbasedFSGNN
from src.utils import (
    set_seeds,
    get_device,
    save_checkpoint,
    get_configs,
    get_used_params,
    find_best_run,
)


device = get_device()

print(device)

with open("sweep_params.yaml") as file:
    sweep_params = yaml.load(file, Loader=yaml.FullLoader)


def run_sweep(c: dict = None):
    set_seeds(seed_no=42)
    global sweep_id

    run = wandb.init(config=c)
    c = wandb.config

    mlp_config, mha_config, data = get_configs(c, device)

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

    existing_best_test_acc, _ = find_best_run(target_dataset=c.dataset["dataset_name"])

    if existing_best_test_acc < trainer.best_test_acc:
        "We find new best model, saving it..."
        used_params = get_used_params(c)
        ckpt = dict(trainer=trainer, params=used_params)
        save_checkpoint(ckpt, data.dataset_name, trainer.best_test_acc, sweep_id)

    wandb.log(data={"test/best_test_acc": trainer.best_test_acc})


sweep_id = wandb.sweep(sweep_params, project="mha-based-fsgnn")
wandb.agent(sweep_id, function=run_sweep)
