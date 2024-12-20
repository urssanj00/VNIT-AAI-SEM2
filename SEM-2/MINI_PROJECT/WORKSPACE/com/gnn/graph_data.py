import torch

edge_index_1 = torch.tensor([
    [0, 1, 2, 3],
    [1, 2, 3, 0]
], dtype=torch.long)
x_1 = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0]
], dtype=torch.float)
y_1 = torch.tensor([0, 1, 0, 1], dtype=torch.long)

edge_index_2 = torch.tensor([
    [0, 1, 2],
    [1, 2, 0]
], dtype=torch.long)
x_2 = torch.tensor([
    [1, 0, 0],
    [0, 1, 1],
    [1, 1, 0]
], dtype=torch.float)
y_2 = torch.tensor([1, 0, 1], dtype=torch.long)

edge_index_3 = torch.tensor([
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 0]
], dtype=torch.long)
x_3 = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1]
], dtype=torch.float)
y_3 = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
