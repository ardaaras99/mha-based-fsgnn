import wandb
import random
import torch
import numpy as np

from pathlib import Path

from src.constants import PROJECT_PATH
from src.dataset import GraphDataset
from src.configurations.model_configs import MHAConfig, MLPConfig


def set_seeds(seed_no: int = 42):
    random.seed(seed_no)
    np.random.seed(seed_no)
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed_all(seed_no)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(ckpt: dict, dataset_name: str, acc, sweep_id):
    test_acc_str = str(round(acc, 5)).replace(".", "_")
    MODELS_DIR = Path.joinpath(
        PROJECT_PATH,
        "model_checkpoints",
        dataset_name,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    SWEEP_ID_FOLDER = MODELS_DIR.joinpath(sweep_id)
    SWEEP_ID_FOLDER.mkdir(parents=True, exist_ok=True)
    FILE_NAME = f"acc:{test_acc_str}-{wandb.run.id}_ckpt.pth"

    ckpt_path = Path.joinpath(MODELS_DIR, SWEEP_ID_FOLDER, FILE_NAME)
    torch.save(ckpt, ckpt_path)


def find_best_run(target_dataset: str):
    # Define the base directory where your model_checkpoints are located
    base_directory = Path.joinpath(PROJECT_PATH, "model_checkpoints")
    highest_accuracy = 0.0
    highest_accuracy_folder = None

    # Iterate through the subfolders of the specified dataset
    dataset_directory = base_directory / target_dataset
    if dataset_directory.is_dir():
        for subfolder in dataset_directory.iterdir():
            # Check if it's a directory
            if subfolder.is_dir():
                # Iterate through files in the subfolder
                for file_path in subfolder.iterdir():
                    file_name = file_path.name
                    if file_name.startswith("acc:"):
                        # Extract the accuracy from the file name
                        parts = file_name.split("-")
                        accuracy_str = parts[0].split("acc:")[1].replace("_", ".")
                        try:
                            accuracy = float(accuracy_str)
                            if accuracy > highest_accuracy:
                                highest_accuracy = accuracy
                                highest_accuracy_folder = subfolder
                                highest_accuracy_file = file_path
                        except ValueError:
                            pass

    # Print the highest accuracy and its corresponding folder
    if highest_accuracy_folder:
        print("Highest Accuracy for", target_dataset, ":", highest_accuracy)
        print("Folder Path:", highest_accuracy_folder)
        print("File Path:", highest_accuracy_file)
    else:
        print(
            "No .pth files with accuracy found for",
            target_dataset,
            "in the directory structure.",
        )

    return highest_accuracy, highest_accuracy_file


def get_configs(c, device):
    dataset_name = c.dataset["dataset_name"]
    data = GraphDataset(dataset_name=dataset_name)
    MASK_MATRIX_CACHE_DIR = (
        Path.cwd() / "mask_matrix_cache" / f"{dataset_name}_mask_matrix_list.pth"
    )

    mask_matrix_list_full = torch.load(MASK_MATRIX_CACHE_DIR)

    L = c.dataset["max_hop"] * 2 + 1
    if dataset_name == "Citeseer":
        #! For Citeseer, we skip power of A_sym since it cause nan loss
        #! We only use A_sym_tilde powers
        mask_matrix_list = mask_matrix_list_full[:L:2]
        L = len(mask_matrix_list)
    else:
        mask_matrix_list = mask_matrix_list_full[:L]

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

    return mlp_config, mha_config, data


def get_used_params(c):
    used_params = {
        "dataset": {
            "dataset_name": c.dataset["dataset_name"],
            "max_hop": c.dataset["max_hop"],
        },
        "mlp": {
            "hidden_dims": c.mlp["hidden_dims"],
            "out_dim": c.mlp["out_dim"],
            "dropout": c.mlp["dropout"],
        },
        "mha": {
            "fan_out": c.mha["fan_out"],
            "p": c.mha["p"],
        },
        "optimizer": {
            "lr": c.optimizer["lr"],
            "weight_decay": c.optimizer["weight_decay"],
        },
        "trainer_pipeline": {
            "max_epochs": c.trainer_pipeline["max_epochs"],
            "patience": c.trainer_pipeline["patience"],
        },
        "skip_connection": c.skip_connection,
    }
    return used_params
