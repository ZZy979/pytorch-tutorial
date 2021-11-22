"""使用邻居采样的连接预测GNN

https://docs.dgl.ai/en/latest/guide/minibatch-link.html
"""
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, EdgeDataLoader
from dgl.dataloading.negative_sampler import Uniform

from gnn.data import RandomGraphDataset
from gnn.dgl.model import GCN, DotProductPredictor, MarginLoss


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gcn = GCN(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, pos_g, neg_g, blocks, x):
        h = self.gcn(blocks, x)
        return self.pred(pos_g, h), self.pred(neg_g, h)


def main():
    data = RandomGraphDataset(100, 500, 10)
    g = data[0]
    train_mask = torch.zeros(g.num_edges(), dtype=torch.bool).bernoulli(0.6)
    train_idx = train_mask.nonzero(as_tuple=True)[0]

    sampler = MultiLayerFullNeighborSampler(2)
    dataloader = EdgeDataLoader(g, train_idx, sampler, negative_sampler=Uniform(5), batch_size=32)

    model = Model(10, 100, 10)
    optimizer = optim.Adam(model.parameters())
    loss_func = MarginLoss()

    for epoch in range(10):
        model.train()
        losses = []
        for input_nodes, pos_g, neg_g, blocks in dataloader:
            pos_score, neg_score = model(pos_g, neg_g, blocks, blocks[0].srcdata['feat'])
            loss = loss_func(pos_score, neg_score)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))


if __name__ == '__main__':
    main()
