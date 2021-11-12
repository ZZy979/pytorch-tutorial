"""在异构图上训练用于顶点分类/回归任务的GNN

https://docs.dgl.ai/en/latest/guide/training-node.html
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import HeteroGraphConv, GraphConv

from gnn.data import UserItemDataset


class RGCN(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats) for rel in rel_names
        }, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats) for rel in rel_names
        }, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


def main():
    data = UserItemDataset()
    g = data[0]
    in_feats = g.nodes['user'].data['feat'].shape[1]
    labels = g.nodes['user'].data['label']
    train_mask = g.nodes['user'].data['train_mask']

    model = RGCN(in_feats, 20, labels.max().item() + 1, g.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(5):
        model.train()
        # forward propagation by using all nodes and extracting the user embeddings
        logits = model(g, g.ndata['feat'])['user']
        # compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # compute validation accuracy, omitted in this example
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == '__main__':
    main()
