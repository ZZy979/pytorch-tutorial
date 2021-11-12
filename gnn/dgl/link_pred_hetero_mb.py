"""异构图上使用邻居采样的连接预测GNN

https://docs.dgl.ai/en/latest/guide/minibatch-link.html
"""
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, EdgeDataLoader
from dgl.dataloading.negative_sampler import Uniform

from gnn.data import UserItemDataset
from gnn.dgl.edge_clf_hetero import HeteroDotProductPredictor
from gnn.dgl.link_pred import compute_loss
from gnn.dgl.node_clf_hetero_mb import RGCN


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, pos_g, neg_g, blocks, x, etype):
        h = self.rgcn(blocks, x)
        return self.pred(pos_g, h, etype), self.pred(neg_g, h, etype)


def main():
    data = UserItemDataset()
    g = data[0]
    in_feats = g.nodes['user'].data['feat'].shape[1]
    train_eids = g.edges['click'].data['train_mask'].nonzero(as_tuple=True)[0]

    sampler = MultiLayerFullNeighborSampler(2)
    dataloader = EdgeDataLoader(
        g, {'click': train_eids}, sampler, negative_sampler=Uniform(5), batch_size=256
    )

    model = Model(in_feats, 20, 5, g.etypes)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        model.train()
        losses = []
        for input_nodes, pos_g, neg_g, blocks in dataloader:
            pos_score, neg_score = model(pos_g, neg_g, blocks, blocks[0].srcdata['feat'], 'click')
            loss = compute_loss(pos_score, neg_score)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))


if __name__ == '__main__':
    main()
