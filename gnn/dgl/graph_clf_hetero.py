"""在异构图上训练用于图分类任务的GNN

参考：<https://docs.dgl.ai/guide/training-graph.html>
"""
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import DGLDataset
from torch.utils.data import DataLoader

from gnn.dgl.graph_clf import collate
from gnn.dgl.node_clf_hetero import RGCN, build_user_item_graph


class HeteroClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feature']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg += dgl.mean_nodes(g, 'h', ntype=ntype)
            return F.softmax(self.classify(hg), dim=0)


class UserItemDataset(DGLDataset):

    def __init__(self):
        self.graphs = []
        self.labels = []
        self.dim_nfeats = 10
        self.gclasses = 3
        super().__init__('user-item')

    def process(self):
        torch.manual_seed(42)
        np.random.seed(42)
        # 随机构造，无实际意义
        for i in range(100):
            g = build_user_item_graph(
                n_users=torch.randint(5, 10, (1,)).item(),
                n_items=torch.randint(5, 10, (1,)).item(),
                n_follows=torch.randint(10, 20, (1,)).item(),
                n_clicks=torch.randint(10, 20, (1,)).item(),
                n_dislikes=torch.randint(10, 20, (1,)).item(),
                n_features=self.dim_nfeats
            )
            self.graphs.append(g)
            self.labels.append(torch.randint(self.gclasses, (1,)))

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


def main():
    dataset = UserItemDataset()
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=collate, drop_last=False
    )

    model = HeteroClassifier(dataset.dim_nfeats, 20, dataset.gclasses, dataset[0][0].etypes)
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
