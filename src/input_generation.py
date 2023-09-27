import torch


def get_symmetric_adjacency(A, self_loop: bool):
    if self_loop:
        A = A + torch.eye(A.shape[0])
    D = torch.diag(torch.sum(A, dim=1))
    D_neg_half = torch.inverse(torch.sqrt(D))
    A_sym = D_neg_half @ A @ D_neg_half
    return A_sym


def get_input_list(A_sym, A_sym_tilde, X: torch.Tensor, k):
    input_list = [X]
    X_a, X_tilde_a = X.clone(), X.clone()

    for _ in range(k):
        X_a = A_sym @ X_a
        X_tilde_a = A_sym_tilde @ X_tilde_a
        input_list.append(X_a)
        input_list.append(X_tilde_a)

    return input_list


def sparse_to_adjacency(edge_index):
    adj_coo = torch.sparse_coo_tensor(
        edge_index, values=torch.ones(edge_index.shape[1])
    )
    A = adj_coo.to_dense()
    return A
