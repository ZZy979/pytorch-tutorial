"""训练用于边分类/回归任务的GNN

https://docs.dgl.ai/en/latest/guide/training-edge.html
"""
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gnn.dgl.node_clf import SAGE


class DotProductPredictor(nn.Module):

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined in node_clf.py
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class MLPPredictor(nn.Module):

    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined in node_clf.py
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)


def main():
    src = np.random.randint(0, 100, 500)
    dst = np.random.randint(0, 100, 500)
    # make it symmetric
    g = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
    # synthetic node and edge features, as well as edge labels
    g.ndata['feature'] = torch.randn(100, 10)
    g.edata['feature'] = torch.randn(1000, 10)
    g.edata['label'] = torch.randn(1000)
    # synthetic train-validation-test splits
    g.edata['train_mask'] = torch.zeros(1000, dtype=torch.bool).bernoulli(0.6)

    node_features = g.ndata['feature']
    edge_label = g.edata['label']
    train_mask = g.edata['train_mask']
    model = Model(10, 20, 5)
    opt = optim.Adam(model.parameters())

    for epoch in range(30):
        pred = model(g, node_features)
        loss = F.mse_loss(pred[train_mask][:, 0], edge_label[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == '__main__':
    main()
