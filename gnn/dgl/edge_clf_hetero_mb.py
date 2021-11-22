"""异构图上使用邻居采样的边分类GNN

https://docs.dgl.ai/en/latest/guide/minibatch-edge.html
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, EdgeDataLoader

from gnn.data import UserItemDataset
from gnn.dgl.model import RGCN, HeteroDotProductPredictor


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, edge_subgraph, blocks, x, etype):
        h = self.rgcn(blocks, x)
        return self.pred(edge_subgraph, h, etype)


def main():
    data = UserItemDataset()
    g = data[0]
    in_feats = g.nodes['user'].data['feat'].shape[1]
    g.edges['click'].data['label'] = g.edges['click'].data['label'].float()
    train_eids = g.edges['click'].data['train_mask'].nonzero(as_tuple=True)[0]

    sampler = MultiLayerFullNeighborSampler(2)
    dataloader = EdgeDataLoader(g, {'click': train_eids}, sampler, batch_size=256)

    model = Model(in_feats, 20, 5, g.etypes)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        model.train()
        losses = []
        for input_nodes, edge_subgraph, blocks in dataloader:
            pred = model(edge_subgraph, blocks, blocks[0].srcdata['feat'], 'click')
            loss = F.mse_loss(pred.squeeze(dim=1), edge_subgraph.edges['click'].data['label'])
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))


if __name__ == '__main__':
    main()
