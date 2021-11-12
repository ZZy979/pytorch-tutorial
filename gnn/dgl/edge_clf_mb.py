"""使用邻居采样的边分类GNN

https://docs.dgl.ai/en/latest/guide/minibatch-edge.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, EdgeDataLoader

from gnn.data import RandomGraphDataset
from gnn.dgl.edge_clf import MLPPredictor
from gnn.dgl.node_clf_mb import GCN


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, num_classes):
        super().__init__()
        self.gcn = GCN(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features, num_classes)

    def forward(self, edge_subgraph, blocks, x):
        h = self.gcn(blocks, x)
        return self.pred(edge_subgraph, h)


def main():
    data = RandomGraphDataset(100, 500, 10)
    g = data[0]
    g.edata['label'] = torch.randint(0, 5, (g.num_edges(),))
    train_mask = torch.zeros(g.num_edges(), dtype=torch.bool).bernoulli(0.6)
    train_idx = train_mask.nonzero(as_tuple=True)[0]

    sampler = MultiLayerFullNeighborSampler(2)
    dataloader = EdgeDataLoader(g, train_idx, sampler, batch_size=32)

    model = Model(10, 20, 10, 5)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        model.train()
        losses = []
        for input_nodes, edge_subgraph, blocks in dataloader:
            logits = model(edge_subgraph, blocks, blocks[0].srcdata['feat'])
            loss = F.cross_entropy(logits, edge_subgraph.edata['label'])
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))


if __name__ == '__main__':
    main()
