# %%
import torch

from src.dataset import GraphDataset, get_mask_matrix_list
from pathlib import Path

dataset_name = "Citeseer"
data = GraphDataset(dataset_name=dataset_name)
data.print_dataset_info()
# %%
#! Can bu dosyayı bir kere runlayınca her şeyi save edecek, sadece sweep functionı içinde readlemek kalacak
#! Bir önceki versiyonda sweep içinde calculate ediyorsun, zaten yapmanı istemediğim şey o, burdan bir kere runladığında yetecek
MAX_HOP = 10
#! max_hop 10 ın ilk 3 elemanı ile max_hop 1 in kendisi aynı, aynı şeyi bir daha hesaplayıp memoryde tutmaya gerek yok
for dataset_name in ["Citeseer"]:
    data = GraphDataset(dataset_name=dataset_name)
    A_sym, A_sym_tilde = data.A_sym, data.A_sym_tilde
    mask_matrix_list = get_mask_matrix_list(A_sym, A_sym_tilde, MAX_HOP)

    PROJECT_FOLDER_DIR = Path.cwd()
    MASK_MATRIX_CACHE_DIR = Path.joinpath(PROJECT_FOLDER_DIR, "mask_matrix_cache")
    MASK_MATRIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FILE_NAME = f"{dataset_name}_mask_matrix_list.pth"
    FILE_PATH = Path.joinpath(MASK_MATRIX_CACHE_DIR, FILE_NAME)
    torch.save(mask_matrix_list, FILE_PATH)

# %%
