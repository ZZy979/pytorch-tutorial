"""训练用于图分类任务的GNN

https://docs.dgl.ai/en/latest/guide/training-graph.html
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import GraphConv
from torch.utils.data import DataLoader


class Classifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels


def main():
    dataset = dgl.data.GINDataset('MUTAG', False)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=collate, drop_last=False
    )

    model = Classifier(dataset.dim_nfeats, 20, dataset.gclasses)
    opt = optim.Adam(model.parameters())

    for epoch in range(20):
        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['attr'].float()
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())


if __name__ == '__main__':
    main()
