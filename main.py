# %%
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch


def sparse_to_adjacency(edge_index):
    adj_coo = torch.sparse_coo_tensor(
        edge_index, values=torch.ones(edge_index.shape[1])
    )
    A = adj_coo.to_dense()
    return A


def load_data(dataset_name, verbose=True):
    dataset = Planetoid(root="datasets/", name=dataset_name)
    dataset.transform = T.NormalizeFeatures()

    data = dataset[0]
    n_feats, n_class = dataset.num_node_features, dataset.num_classes

    # TODO: replace this with logger, check end-to-end nlp video for this
    if verbose:
        print(f"Number of Classes in {dataset_name}:", dataset.num_classes)
        print(f"Number of Node Features in {dataset_name}:", dataset.num_node_features)
    return data, n_feats, n_class


data, n_feats, n_class = load_data(dataset_name="Cora")
A = sparse_to_adjacency(data.edge_index)
X = data.x
