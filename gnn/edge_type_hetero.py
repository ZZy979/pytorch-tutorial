"""在异构图上训练用于预测边类型任务的GNN

参考：<https://docs.dgl.ai/guide/training-edge.html>
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gnn.node_clf_hetero import RGCN, build_user_item_graph


class HeteroMLPPredictor(nn.Module):

    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        score = self.W(torch.cat([edges.src['h'], edges.dst['h']], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations for each edge type computed from node_clf_hetero.py
        with graph.local_scope():
            graph.ndata['h'] = h  # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, out_classes, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroMLPPredictor(out_features, out_classes)

    def forward(self, g, x, dec_graph):
        h = self.rgcn(g, x)
        return self.pred(dec_graph, h)


def main():
    g = build_user_item_graph()
    user_feats = g.nodes['user'].data['feature']
    item_feats = g.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    dec_graph = g['user', :, 'item']
    edge_label = dec_graph.edata[dgl.ETYPE]
    edge_label -= edge_label.min().item()

    model = Model(user_feats.shape[1], 20, 5, 2, g.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(10):
        logits = model(g, node_features, dec_graph)
        loss = F.cross_entropy(logits, edge_label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == '__main__':
    main()
