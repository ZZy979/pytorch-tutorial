"""在异构图上训练用于边分类/回归任务的GNN

https://docs.dgl.ai/en/latest/guide/training-edge.html
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gnn.data import UserItemDataset
from gnn.dgl.node_clf_hetero import RGCN


class HeteroDotProductPredictor(nn.Module):

    def forward(self, graph, h, etype):
        # h contains the node representations for each edge type computed from node_clf_hetero.py
        with graph.local_scope():
            graph.ndata['h'] = h  # assigns 'h' of all node types in one shot
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class HeteroMLPPredictor(nn.Module):

    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        score = self.W(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
        return {'score': score}

    def forward(self, graph, h, etype):
        # h contains the node representations for each edge type computed from node_clf_hetero.py
        with graph.local_scope():
            graph.ndata['h'] = h  # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_features, hidden_features, out_features, rel_names)
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
