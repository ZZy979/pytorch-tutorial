"""训练用于顶点分类/回归任务的GNN

https://docs.dgl.ai/en/latest/guide/training-node.html
"""
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import CiteseerGraphDataset

from gnn.dgl.model import SAGEFull
from gnn.utils import accuracy


def main():
    # load data
    dataset = CiteseerGraphDataset()
    g = dataset[0]

    node_features = g.ndata['feat']
    node_labels = g.ndata['label']
    n_features = node_features.shape[1]
    n_labels = dataset.num_classes

    # get split masks
    train_mask = g.ndata['train_mask']
    valid_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    model = SAGEFull(in_feats=n_features, hid_feats=100, out_feats=n_labels)
    opt = optim.Adam(model.parameters())

    for epoch in range(30):
        model.train()
        # forward propagation by using all nodes
        logits = model(g, node_features)
        # compute loss
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        # compute validation accuracy
        acc = accuracy(logits[valid_mask], node_labels[valid_mask])
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('Epoch {:d} | Loss {:.4f} | Accuracy {:.2%}'.format(epoch + 1, loss.item(), acc))
    acc = accuracy(model(g, node_features)[test_mask], node_labels[test_mask])
    print('Test accuracy {:.4f}'.format(acc))


if __name__ == '__main__':
    main()
