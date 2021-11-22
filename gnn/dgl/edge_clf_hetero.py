"""在异构图上训练用于边分类/回归任务的GNN

https://docs.dgl.ai/en/latest/guide/training-edge.html
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gnn.data import UserItemDataset
from gnn.dgl.model import RGCNFull, HeteroDotProductPredictor


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.rgcn = RGCNFull(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, x, etype):
        h = self.rgcn(g, x)
        return self.pred(g, h, etype)


def main():
    data = UserItemDataset()
    g = data[0]
    in_feats = g.nodes['user'].data['feat'].shape[1]
    labels = g.edges['click'].data['label'].float()
    train_mask = g.edges['click'].data['train_mask']

    model = Model(in_feats, 20, 5, g.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(10):
        pred = model(g, g.ndata['feat'], 'click')
        loss = F.mse_loss(pred[train_mask].squeeze(dim=1), labels[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == '__main__':
    main()
