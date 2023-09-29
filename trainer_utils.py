import os
import re
import wandb
import torch
from pathlib import Path


def str_to_acc(filename: str) -> float:
    pattern = r"(\d+_\d+)"
    match = re.search(pattern, filename)
    if match:
        extracted_value = match.group(1)
        float_value = float(extracted_value.replace("_", "."))
        return float_value


def to_device(data, target, device):
    return data.to(device), target.to(device)


def log_train_parameters(loss: float, lr: float, fracs_d: dict, epoch: int):
    wandb.log(data={"train/loss": loss}, step=epoch)
    wandb.log(data={"train/lr": lr}, step=epoch)

    try:
        for k, v in fracs_d.items():
            order1, order2 = v
            wandb.log(data={f"train/{k}_order1": order1}, step=epoch)
            wandb.log(data={f"train/{k}_order2": order2}, step=epoch)

    except AttributeError:
        print("Model does not have frac_order parameter")


def log_test_parameters(test_acc: float, epoch: int):
    wandb.log(data={"test/test_accuracy": test_acc}, step=epoch)


def log_model_artifact(test_acc: str, ckpt_file: Path):
    artifact_name = f"{wandb.run.id}_{ckpt_file.stem}_{test_acc}"
    artifact = wandb.Artifact(name=artifact_name, type="Model")
    artifact.add_file(ckpt_file)
    wandb.log_artifact(artifact)


def get_existing_model(model_name: str, SAVE_DIR: Path):
    files_in_dir = os.listdir(SAVE_DIR)

    for file in files_in_dir:
        if file.startswith(model_name):
            best_acc = str_to_acc(filename=file)
            best_model = torch.load(SAVE_DIR / file)
            return best_acc, best_model


import copy


class EarlyStopping:
    def __init__(self, patience: int, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_test_acc = 0
        self.best_model = None
        self.early_stop = False

    def __call__(self, test_acc: float, model: torch.nn.Module, epoch: int):
        if test_acc > self.best_test_acc:
            self.counter = 0
            self.best_test_acc = test_acc
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping at epoch {epoch}")

        return self.best_test_acc, self.best_model
