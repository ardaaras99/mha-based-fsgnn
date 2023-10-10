# %%
import torch


import torch


def custom_reciprocal(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.where(torch.abs(x) < eps, 0, torch.reciprocal(x))


if __name__ == "__main__":
    a = torch.tensor([1, 2, 0, 3, 0])
    b = custom_reciprocal(a)
    print(a)
    print(b)
    diag_matrix = torch.diag(b)
    print(diag_matrix)


# %%
