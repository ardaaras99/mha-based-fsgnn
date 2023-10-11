# %%
import torch
from src.utils import find_best_run
from src.trainer import Trainer

#! This file enable us to reproduce the results of best_model

dataset_name = "Citeseer"
ckpt_folder_path, ckpt_file_path = find_best_run(target_dataset=dataset_name)
ckpt = torch.load(ckpt_file_path)

trainer: Trainer = ckpt["trainer"]
params = ckpt["params"]

trainer.model = trainer.best_model
train_acc, test_acc = trainer.eval_model(trainer.data)

print("Test Accuracy from loaded model:", test_acc)
print("Best test accuracy:", trainer.best_test_acc)

# %%
