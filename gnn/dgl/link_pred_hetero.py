"""在异构图上训练用于连接预测任务的GNN

https://docs.dgl.ai/en/latest/guide/training-link.html
"""
import dgl
import torch
import torch.nn as nn
import torch.optim as optim

from gnn.data import UserItemDataset
from gnn.dgl.edge_clf_hetero import HeteroDotProductPredictor
from gnn.dgl.link_pred import compute_loss
from gnn.dgl.node_clf_hetero import RGCN


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes}
    )


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, neg_g, x, etype):
        h = self.rgcn(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


def main():
    data = UserItemDataset()
    g = data[0]
    in_feats = g.nodes['user'].data['feat'].shape[1]
    k = 5

    model = Model(in_feats, 20, 5, g.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(10):
        negative_graph = construct_negative_graph(g, k, ('user', 'click', 'item'))
        pos_score, neg_score = model(
            g, negative_graph, g.ndata['feat'], ('user', 'click', 'item')
        )
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == '__main__':
    main()
