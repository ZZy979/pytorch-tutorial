import argparse

import dgl
import torch
import torch.nn.functional as F

from pytorch_tutorial.gnn.gat.model import GAT
from pytorch_tutorial.gnn.utils import load_dataset, accuracy, evaluate_accuracy


def train(args):
    data = load_dataset(args.dataset)
    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_classes

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # create model
    heads = [args.num_heads] * args.num_layers + [args.num_out_heads]
    model = GAT(
        g, args.num_layers, num_feats, args.num_hidden, n_classes, heads, F.elu,
        args.in_drop, args.attn_drop, args.negative_slope, args.residual
    )
    print(model)
    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    for epoch in range(args.epochs):
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = evaluate_accuracy(model, labels, val_mask, features)
        print('Epoch {:05d} | Loss {:.4f} | TrainAcc {:.4f} |  ValAcc {:.4f}'.format(
            epoch, loss.item(), train_acc, val_acc
        ))

    print()
    acc = evaluate_accuracy(model, labels, test_mask, features)
    print('Test Accuracy {:.4f}'.format(acc))


def main():
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed'], default='cora', help='dataset')
    parser.add_argument('--gpu', type=int, default=-1, help='which GPU to use, set -1 to use CPU')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--num-heads', type=int, default=8, help='number of hidden attention heads')
    parser.add_argument('--num-out-heads', type=int, default=1,
                        help='number of output attention heads')
    parser.add_argument('--num-layers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--num-hidden', type=int, default=8, help='number of hidden units')
    parser.add_argument('--in-drop', type=float, default=.6, help='input feature dropout')
    parser.add_argument('--attn-drop', type=float, default=.6, help='attention dropout')
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help='the negative slope of leaky relu')
    parser.add_argument('--residual', action='store_true', default=False,
                        help='use residual connection')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
