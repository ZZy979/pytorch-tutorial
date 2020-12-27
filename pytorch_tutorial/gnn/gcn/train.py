import argparse

import dgl
import torch.nn as nn
import torch.optim as optim

from pytorch_tutorial.gnn.gcn.model import GCN
from pytorch_tutorial.gnn.utils import load_dataset, evaluate_accuracy


def train(args):
    data = load_dataset(args.dataset)
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    num_classes = data.num_classes

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    model = GCN(num_feats, args.num_hidden, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(args.epochs):
        model.train()
        logits = model(g, features)
        loss = criterion(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate_accuracy(model, labels, val_mask, g, features)
        print('Epoch {:05d} | Loss {:.4f} | ValAcc {:.4f}'.format(epoch, loss.item(), acc))
    acc = evaluate_accuracy(model, labels, test_mask, g, features)
    print('Test Accuracy {:.4f}'.format(acc))


def main():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed'], default='cora', help='dataset')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--num-hidden', type=int, default=16, help='number of hidden units')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()