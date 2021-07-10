import argparse

import dgl

from gnn.lp.model import LabelPropagation
from gnn.utils import load_citation_dataset, accuracy


def train(args):
    data = load_citation_dataset(args.dataset)
    g = data[0]
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']

    g = dgl.add_self_loop(dgl.remove_self_loop(g))

    lp = LabelPropagation(args.num_layers, args.alpha)
    logits = lp(g, labels, train_mask)
    test_acc = accuracy(logits[test_mask], labels[test_mask])
    print('Test Accuracy {:.4f}'.format(test_acc))


def main():
    parser = argparse.ArgumentParser(description='Label Propagation')
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed'], default='cora', help='dataset')
    parser.add_argument('--num-layers', type=int, default=10, help='number of propagation layers')
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
