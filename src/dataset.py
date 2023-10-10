# %%
from pathlib import Path
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

CURRENT_DIR = Path(__file__).parent


class GraphDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.root_dir = Path.joinpath(CURRENT_DIR.parent, "datasets")
        self._set_variables()

    def _set_variables(self):
        dataset = Planetoid(
            root=self.root_dir, name=self.dataset_name, transform=T.NormalizeFeatures()
        )
        self.dataset = dataset
        data = dataset[0]

        self.n_feats, self.n_class = dataset.num_node_features, dataset.num_classes

        self.X = data.x
        self.A = sparse_to_adjacency(data.edge_index)
        self.A_sym = get_symmetric_adjacency(self.A, self_loop=False)
        self.A_sym_tilde = get_symmetric_adjacency(self.A, self_loop=True)

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.y = data.y

    def print_dataset_info(self):
        print(f"Number of Classes in {self.dataset_name}:", self.n_class)
        print(f"Number of Node Features in {self.dataset_name}:", self.n_feats)


# %%
# Utility functions


def custom_reciprocal(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.where(torch.abs(x) < eps, 0, torch.reciprocal(x))


def get_symmetric_adjacency(A, self_loop: bool):
    if self_loop:
        A = A + torch.eye(A.shape[0])
    D = torch.diag(torch.sum(A, dim=1))
    D = torch.sqrt(D)
    D_neg_half = custom_reciprocal(D)
    A_sym = D_neg_half @ A @ D_neg_half
    return A_sym


def sparse_to_adjacency(edge_index):
    adj_coo = torch.sparse_coo_tensor(
        edge_index, values=torch.ones(edge_index.shape[1])
    )
    A = adj_coo.to_dense()
    return A


def powers_of_A(A, k: int, type: str = "sym") -> dict:
    A_dict = {}
    A_dict[f"A_{type}^1"] = A
    for i in range(2, k + 1):
        A_dict[f"A_{type}^{i}"] = A_dict[f"A_{type}^{i-1}"] @ A
    return A_dict


def get_mask_matrix_list(A_sym, A_sym_tilde, max_hop: int):
    d1 = powers_of_A(A_sym, k=max_hop, type="sym")
    d2 = powers_of_A(A_sym_tilde, k=max_hop, type="sym_tilde")

    mask_matrix_list = [torch.eye(A_sym.shape[0])]
    for (k1, A_sym), (k2, A_sym_tilde) in zip(d1.items(), d2.items()):
        mask_matrix_list.append(A_sym)
        mask_matrix_list.append(A_sym_tilde)
    return mask_matrix_list


# def get_input_list(A_sym, A_sym_tilde, X: torch.Tensor, k):
#     input_list = [X]
#     X_a, X_tilde_a = X.clone(), X.clone()

#     for _ in range(k):
#         X_a = A_sym @ X_a
#         X_tilde_a = A_sym_tilde @ X_tilde_a
#         input_list.append(X_a)
#         input_list.append(X_tilde_a)

#     return input_list
