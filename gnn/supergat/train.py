import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import gnn_benckmark
from ogb.nodeproppred import DglNodePropPredDataset

from gnn.supergat.model import SuperGAT
from gnn.utils import load_citation_dataset, split_idx, set_random_seed, accuracy

DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn-arxiv',
    'cora_full', 'cs', 'physics', 'photo', 'computers'
]


def load_data(name, seed):
    if name == 'ogbn-arxiv':
        data = DglNodePropPredDataset('ogbn-arxiv', 'D:\\ogb')
        g, labels = data[0]
        split = data.get_idx_split()
        return g, labels.squeeze(dim=-1), data.num_classes, \
               split['train'], split['valid'], split['test']
    elif name in ('cora', 'citeseer', 'pubmed'):
        data = load_citation_dataset(name)
    elif name == 'cora_full':
        data = gnn_benckmark.CoraFullDataset()
    elif name in ('cs', 'physics'):
        data = gnn_benckmark.Coauthor(name)
    elif name in ('photo', 'computers'):
        data = gnn_benckmark.AmazonCoBuy(name)
    else:
        raise ValueError('Unknown dataset:', name)

    g = data[0]
    # https://github.com/dmlc/dgl/issues/2479
    num_classes = data.num_classes
    if name in ('photo', 'computers'):
        num_classes = g.ndata['label'].max().item() + 1
    if 'train_mask' in g.ndata:
        train_idx = g.ndata['train_mask'].nonzero(as_tuple=True)[0]
        val_idx = g.ndata['val_mask'].nonzero(as_tuple=True)[0]
        test_idx = g.ndata['test_mask'].nonzero(as_tuple=True)[0]
    else:
        train_idx, val_idx, test_idx = split_idx(torch.arange(g.num_nodes()), 0.2, 0.3, seed)
    return g, g.ndata['label'], num_classes, train_idx, val_idx, test_idx


def train(args):
    set_random_seed(args.seed)
    g, labels, num_classes, train_idx, val_idx, test_idx = load_data(args.dataset, args.seed)
    features = g.ndata['feat']

    model = SuperGAT(
        features.shape[1], args.num_hidden, num_classes, args.num_heads, args.attn_type,
        args.neg_sample_ratio, 0, args.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits, attn_loss = model(g, features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss += args.attn_loss_weight * attn_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx], labels[train_idx])
        val_acc = evaluate(model, g, features, labels, val_idx)
        print('Epoch {:04d} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, loss.item(), train_acc, val_acc
        ))
    acc = evaluate(model, g, features, labels, test_idx)
    print('Test Accuracy {:.4f}'.format(acc))


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(g, features)
    return accuracy(logits[mask], labels[mask])


def main():
    parser = argparse.ArgumentParser(description='SuperGAT')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', choices=DATASETS, default='cora', help='dataset')
    parser.add_argument('--num-hidden', type=int, default=8, help='number of hidden units')
    parser.add_argument('--num-heads', type=int, default=8, help='number of attention heads')
    parser.add_argument(
        '--attn-type', choices=['GO', 'DP', 'SD', 'MX'], default='MX', help='type of attention'
    )
    parser.add_argument(
        '--neg-sample-ratio', type=float, default=0.8,
        help='ratio of the number of sampled negative edges to the number of positive edges'
    )
    parser.add_argument('--dropout', type=float, default=0.6, help='attention dropout probability')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--attn-loss-weight', type=float, default=2.0, help='attention loss weight')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
