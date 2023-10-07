import torch
from src.dataset import get_mask_matrix_list

data = GraphDataset(dataset_name=c.dataset["dataset_name"])

MAX_HOP = 8

mask_matrix_lists = {}

for max_hop in range(1, MAX_HOP+1):
  A_sym, A_sym_tilde = # generate matrices
  mask_matrix_list = get_mask_matrix_list(A_sym, A_sym_tilde, max_hop)
  mask_matrix_lists[max_hop] = mask_matrix_list

torch.save(mask_matrix_lists, 'mask_matrix_lists.pt')