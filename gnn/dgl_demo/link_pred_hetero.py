"""在异构图上训练用于连接预测任务的GNN

参考：<https://docs.dgl.ai/guide/training-link.html>
"""
import dgl
import torch
import torch.nn as nn
import torch.optim as optim

from gnn.dgl_demo.edge_clf_hetero import HeteroDotProductPredictor
from gnn.dgl_demo.link_pred import compute_loss
from gnn.dgl_demo.node_clf_hetero import RGCN, build_user_item_graph


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
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


def main():
    hetero_graph = build_user_item_graph()
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    k = 5

    model = Model(user_feats.shape[1], 20, 5, hetero_graph.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(10):
        negative_graph = construct_negative_graph(hetero_graph, k, ('user', 'click', 'item'))
        pos_score, neg_score = model(
            hetero_graph, negative_graph, node_features, ('user', 'click', 'item')
        )
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == '__main__':
    main()
