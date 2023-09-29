# %%
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

from src.dataset import GraphDataset
from src.input_generation import get_mask_matrix_list
from src.model import MHAbasedFSGNN
from src.model_configs import MLPConfig, MHAConfig


data = GraphDataset(dataset_name="Cora")
data.print_dataset_info()

max_hop = 4
L = 2 * max_hop + 1

mask_matrix_list = get_mask_matrix_list(data.A_sym, data.A_sym_tilde, max_hop=max_hop)


#! Since we use skip connection, chose L wisely so that head concatenation will not reduce dimension
# for max_hop = 3, L = 7, if we choose fan_in = 64, we loose dimension

mlp_config = MLPConfig(in_dim=data.n_feats, hidden_dims=300, out_dim=90)
mha_config = MHAConfig(
    fan_in=90, fan_out=90, n_heads=L, p=0.6, mask_matrix_list=mask_matrix_list
)

model = MHAbasedFSGNN(
    mlp_config=mlp_config, mha_config=mha_config, n_class=data.n_class
)
# %%

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

best_test_acc = 0
t = tqdm(range(200))
for epoch in t:
    model.train()
    out = model(data.X)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    model.eval()
    _, pred = model(data.X).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()

    train_acc = (
        float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
        / data.train_mask.sum().item()
    )
    if acc > best_test_acc:
        best_test_acc = acc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    t.set_description(
        f"Loss: {loss:.4f}, Best Test Acc: {best_test_acc:.3f}, Train Acc: {train_acc:.3f}"
    )
# %%
