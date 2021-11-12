"""训练用于连接预测任务的GNN

https://docs.dgl.ai/en/latest/guide/training-link.html
"""
import dgl
import torch
import torch.nn as nn
import torch.optim as optim

from gnn.data import RandomGraphDataset
from gnn.dgl.edge_clf import DotProductPredictor
from gnn.dgl.node_clf import SAGE


def construct_negative_graph(graph, k):
    src, dst = graph.edges()
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.number_of_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.number_of_nodes())


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, g, neg_g, x):
        h = self.sage(g, x)
        return self.pred(g, h), self.pred(neg_g, h)


def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()


def main():
    data = RandomGraphDataset(100, 500, 10)
    g = data[0]
    node_features = g.ndata['feat']
    n_features = node_features.shape[1]
    k = 5

    model = Model(n_features, 100, 100)
    opt = optim.Adam(model.parameters())

    for epoch in range(10):
        negative_graph = construct_negative_graph(g, k)
        pos_score, neg_score = model(g, negative_graph, node_features)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == '__main__':
    main()
