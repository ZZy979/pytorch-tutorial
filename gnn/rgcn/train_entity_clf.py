import argparse

import torch.nn.functional as F
import torch.optim as optim

from gnn.rgcn.model_hetero import EntityClassification
from gnn.utils import load_rdf_dataset, accuracy


def train(args):
    data = load_rdf_dataset(args.dataset)
    g = data[0]
    category = data.predict_category
    num_classes = data.num_classes
    labels = g.nodes[category].data['labels']
    train_mask = g.nodes[category].data['train_mask'].bool()
    test_mask = g.nodes[category].data['test_mask'].bool()

    model = EntityClassification(
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes},
        args.num_hidden, num_classes, list(set(g.etypes)),
        args.num_hidden_layers, args.num_bases, args.self_loop, args.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits = model(g)[category]
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        print('Epoch {:04d} | Loss {:.4f} | Train Acc {:.4f}'.format(epoch, loss.item(), train_acc))

    test_acc = accuracy(model(g)[category][test_mask], labels[test_mask])
    print('Test Accuracy {:.4f}'.format(test_acc))


def main():
    parser = argparse.ArgumentParser(description='R-GCN Entity Classification')
    parser.add_argument(
        '--dataset', choices=['aifb', 'mutag', 'bgs', 'am'], default='aifb', help='dataset'
    )
    parser.add_argument('--num-hidden', type=int, default=16, help='number of hidden units')
    parser.add_argument('--num-hidden-layers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--num-bases', type=int, default=0)
    parser.add_argument('--self-loop', action='store_true', help='include self-loop message')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
