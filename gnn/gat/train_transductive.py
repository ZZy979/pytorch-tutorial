import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim

from gnn.gat.model import GAT
from gnn.utils import load_citation_dataset, accuracy


def train(args):
    data = load_citation_dataset(args.dataset)
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    num_heads = [args.num_heads] * (args.num_layers - 1) + [args.num_out_heads]
    model = GAT(features.shape[1], args.num_hidden, data.num_classes, num_heads, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = evaluate(model, g, features, labels, val_mask)
        print('Epoch {:04d} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, loss.item(), train_acc, val_acc
        ))

    print()
    acc = evaluate(model, g, features, labels, test_mask)
    print('Test Accuracy {:.4f}'.format(acc))


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    return accuracy(logits[mask], labels[mask])


def main():
    parser = argparse.ArgumentParser(description='GAT Transductive Training')
    parser.add_argument(
        '--dataset', choices=['cora', 'citeseer', 'pubmed'], default='cora', help='dataset'
    )
    parser.add_argument('--num-layers', type=int, default=2, help='number of GAT layers')
    parser.add_argument('--num-hidden', type=int, default=8, help='number of hidden units')
    parser.add_argument(
        '--num-heads', type=int, default=8, help='number of attention heads in hidden layers'
    )
    parser.add_argument(
        '--num-out-heads', type=int, default=1, help='number of attention heads in output layer'
    )
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
