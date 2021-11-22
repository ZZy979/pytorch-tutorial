"""在异构图上训练用于顶点分类/回归任务的GNN

https://docs.dgl.ai/en/latest/guide/training-node.html
"""
import torch.nn.functional as F
import torch.optim as optim

from gnn.data import UserItemDataset
from gnn.dgl.model import RGCNFull


def main():
    data = UserItemDataset()
    g = data[0]
    in_feats = g.nodes['user'].data['feat'].shape[1]
    labels = g.nodes['user'].data['label']
    train_mask = g.nodes['user'].data['train_mask']

    model = RGCNFull(in_feats, 20, labels.max().item() + 1, g.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(10):
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
