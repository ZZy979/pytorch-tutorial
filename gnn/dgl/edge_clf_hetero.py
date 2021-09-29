"""在异构图上训练用于边分类/回归任务的GNN

https://docs.dgl.ai/en/latest/guide/training-edge.html
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gnn.dgl.node_clf_hetero import RGCN, build_user_item_graph


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
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
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
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype)


def main():
    g = build_user_item_graph()
    user_feats = g.nodes['user'].data['feature']
    item_feats = g.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    labels = g.edges['click'].data['label'].float()
    train_mask = g.edges['click'].data['train_mask']

    model = Model(user_feats.shape[1], 20, 5, g.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(10):
        pred = model(g, node_features, 'click')
        loss = F.mse_loss(pred[train_mask][:, 0], labels[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == '__main__':
    main()
