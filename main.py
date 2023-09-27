# %%
from dataset import GraphDataset
import torch
from models import Model
from tqdm.auto import tqdm
import torch.nn.functional as F

data = GraphDataset(dataset_name="Cora")
data.print_dataset_info()

max_hop = 3
L = 2 * max_hop + 1

data.set_input_list(max_hop=max_hop)
print(len(data.input_list))


model = Model(fan_in=data.n_feats, fan_middle=210, n_heads=L, n_class=data.n_class)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

best_test_acc = 0
t = tqdm(range(200))
for epoch in t:
    model.train()
    out = model(data.input_list)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    model.eval()
    _, pred = model(data.input_list).max(dim=1)
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
