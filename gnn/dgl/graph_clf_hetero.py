"""在异构图上训练用于图分类任务的GNN

https://docs.dgl.ai/en/latest/guide/training-graph.html
"""
import random

import dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

from gnn.data import UserItemDataset as UserItemOneGraphDataset
from gnn.dgl.model import RGCNFull


class HeteroClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        self.rgcn = RGCNFull(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = self.rgcn(g, g.ndata['feat'])
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = sum(dgl.mean_nodes(g, 'h', ntype=ntype) for ntype in g.ntypes)
            return F.softmax(self.classify(hg), dim=0)


class UserItemDataset(DGLDataset):

    def __init__(self):
        self.graphs = []
        self.labels = []
        self.n_features = 10
        self.num_classes = 3
        super().__init__('user-item')

    def process(self):
        random.seed(42)
        # 随机构造，无实际意义
        for i in range(100):
            g = UserItemOneGraphDataset(
                n_users=random.randint(5, 10),
                n_items=random.randint(5, 10),
                n_follows=random.randint(10, 20),
                n_clicks=random.randint(10, 20),
                n_dislikes=random.randint(10, 20),
                n_features=self.n_features
            )[0]
            self.graphs.append(g)
            self.labels.append(random.randrange(self.num_classes))

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


def main():
    dataset = UserItemDataset()
    dataloader = GraphDataLoader(dataset, batch_size=32, shuffle=True)

    model = HeteroClassifier(dataset.n_features, 20, dataset.num_classes, dataset[0][0].etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(5):
        for batched_graph, labels in dataloader:
            logits = model(batched_graph)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())


if __name__ == '__main__':
    main()
