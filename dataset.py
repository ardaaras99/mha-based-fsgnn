from pathlib import Path
import torch
from src.input_generation import (
    get_symmetric_adjacency,
    get_input_list,
    sparse_to_adjacency,
)

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

CURRENT_DIR = Path(__file__).parent


class GraphDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.root_dir = Path.joinpath(CURRENT_DIR, dataset_name)
        self._set_variables()

    def _set_variables(self):
        dataset = Planetoid(
            root=self.root_dir, name=self.dataset_name, transform=T.NormalizeFeatures()
        )
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

    def set_input_list(self, max_hop):
        """
        This function sets the input list for the GCN model.
        """
        self.input_list = get_input_list(self.A_sym, self.A_sym_tilde, self.X, max_hop)
