import argparse

import dgl
import torch.nn.functional as F
import torch.optim as optim

from pytorch_tutorial.gnn.gat.model import GAT
from pytorch_tutorial.gnn.utils import load_citation_dataset, accuracy


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

    model = GAT(features.shape[1], 8, data.num_classes, [8, args.num_out_heads])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        print('Epoch {:04d} | Loss {:.4f} | TrainAcc {:.4f} | ValAcc {:.4f}'.format(
            epoch, loss.item(), train_acc, val_acc
        ))

    print()
    acc = accuracy(model(g, features)[test_mask], labels[test_mask])
    print('Test Accuracy {:.4f}'.format(acc))


def main():
    parser = argparse.ArgumentParser(description='GAT Transductive Training')
    parser.add_argument(
        '--dataset', choices=['cora', 'citeseer', 'pubmed'], default='cora', help='dataset'
    )
    parser.add_argument(
        '--num-out-heads', type=int, default=1, help='number of attention heads in output layer'
    )
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
