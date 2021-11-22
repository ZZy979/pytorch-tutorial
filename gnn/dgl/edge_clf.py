"""训练用于边分类/回归任务的GNN

https://docs.dgl.ai/en/latest/guide/training-edge.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gnn.data import RandomGraphDataset
from gnn.dgl.model import SAGEFull, DotProductPredictor


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGEFull(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)


def main():
    data = RandomGraphDataset(100, 500, 10)
    g = data[0]
    # synthetic edge labels
    g.edata['label'] = torch.randn(1000)
    # synthetic train-validation-test splits
    g.edata['train_mask'] = torch.zeros(1000, dtype=torch.bool).bernoulli(0.6)

    edge_label = g.edata['label']
    train_mask = g.edata['train_mask']
    model = Model(10, 20, 5)
    opt = optim.Adam(model.parameters())

    for epoch in range(30):
        pred = model(g, g.ndata['feat'])
        loss = F.mse_loss(pred[train_mask][:, 0], edge_label[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == '__main__':
    main()
