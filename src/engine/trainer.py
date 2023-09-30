from tqdm.auto import trange
import torch
from pathlib import Path
import wandb
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import re
import wandb
import torch
from pathlib import Path
import copy
from src.dataset import GraphDataset


class Trainer:
    def __init__(
        self,
        # TODO: add other possible model names here as typing
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data: GraphDataset,
    ):
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.checkpoint = {}

    def pipeline(
        self,
        max_epochs: int,
        patience: int,
        wandb_flag: bool = False,
        sweep_id: str = None,
        early_stop_verbose: bool = False,
    ):
        early_stopping = EarlyStopping(patience=patience, verbose=early_stop_verbose)

        t = tqdm(range(max_epochs))
        for epoch in t:
            self.model.train()
            e_loss = self.train_epoch(data_key="train")

            train_acc, test_acc = self.eval_model(self.data)

            if wandb_flag:
                self.epoch_wandb_log(loss=e_loss, test_acc=test_acc, epoch=epoch)

            best_test_acc, _ = early_stopping(test_acc, self.model, epoch)
            t.set_description(
                f"Loss: {e_loss:.4f}, Best Test Acc: {best_test_acc:.3f}, Train Acc: {train_acc:.3f}"
            )
            if early_stopping.early_stop:
                break

        # self.pipeline_wandb_log(best_test_acc, best_model, sweep_id)

    def train_epoch(self, data_key: str = "train"):
        loss = 0
        # TODO: replace string check with enum
        self.model.train()
        out = self.model(self.data.X)
        loss = F.nll_loss(out[self.data.test_mask], self.data.y[self.data.test_mask])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def eval_model(self, data: GraphDataset):
        with torch.no_grad():
            self.model.eval()
            _, pred = self.model(self.data.X).max(dim=1)

            test_acc = (
                float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                / data.test_mask.sum().item()
            )

            train_acc = (
                float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
                / data.train_mask.sum().item()
            )

            return 100.0 * train_acc, 100.0 * test_acc

    def test_batch(self):
        pass

    def train_batch(self):
        #! for graph datasets, we do not use batch approach currently
        pass

    def save_checkpoint(self, test_acc: float, sweep_id: str):
        test_acc = str(round(test_acc, 5)).replace(".", "_")

        MODELS_DIR = Path(__file__).parent.parent.joinpath("model_checkpoints")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        SWEEP_ID_FOLDER = MODELS_DIR.joinpath(sweep_id)
        SWEEP_ID_FOLDER.mkdir(parents=True, exist_ok=True)
        FILE_NAME = f"{self.model.model_name}-{test_acc}-{wandb.run.id}-ckpt.pth"

        ckpt_path = Path.joinpath(MODELS_DIR, SWEEP_ID_FOLDER, FILE_NAME)
        torch.save(self.checkpoint, ckpt_path)

    def epoch_wandb_log(self, loss, lr, fracs_d, epoch, test_acc):
        log_train_parameters(loss=loss, lr=lr, fracs_d=fracs_d, epoch=epoch)
        log_test_parameters(test_acc=test_acc, epoch=epoch)

    def pipeline_wandb_log(
        self, best_test_acc, best_model: torch.nn.Module, sweep_id: str
    ):
        self.checkpoint["best_test_acc"] = best_test_acc
        self.checkpoint["model"] = best_model
        self.checkpoint["model_state_dict"] = best_model.state_dict()
        self.save_checkpoint(best_test_acc, sweep_id)
        wandb.log(data={"test/best_test_acc": best_test_acc})

    # def sanity_check(self, best_model, best_test_acc):
    #     """
    #     A sanity check for the test accuracy of the best model, use it if you want to be sure
    #     """
    #     self.model = copy.deepcopy(best_model)
    #     acc_new = self.test(data_key="test")
    #     print(f"Best test accs: {best_test_acc}, {acc_new}")


# %%

## Utility functions for Trainer


def str_to_acc(filename: str) -> float:
    pattern = r"(\d+_\d+)"
    match = re.search(pattern, filename)
    if match:
        extracted_value = match.group(1)
        float_value = float(extracted_value.replace("_", "."))
        return float_value


def to_device(data, target, device):
    return data.to(device), target.to(device)


def log_train_parameters(loss: float, lr: float, epoch: int):
    wandb.log(data={"train/loss": loss}, step=epoch)
    wandb.log(data={"train/lr": lr}, step=epoch)


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
