from tqdm.auto import trange
import torch
from pathlib import Path
import wandb
import torch.nn.functional as F
from tqdm.auto import tqdm

from trainer_utils import log_train_parameters, log_test_parameters, EarlyStopping
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
        e_loss = 0
        # TODO: replace string check with enum
        self.model.train()
        out = self.model(self.data.X)
        e_loss = F.nll_loss(
            out[self.data.train_mask], self.data.y[self.data.train_mask]
        )
        self.optimizer.zero_grad()
        e_loss.backward()
        self.optimizer.step()
        return e_loss

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
