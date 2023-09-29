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
