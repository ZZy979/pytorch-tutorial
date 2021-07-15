import argparse

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset

from gnn.cs.model import MLP, CorrectAndSmooth
from gnn.utils import set_random_seed, get_device, load_citation_dataset, accuracy


def load_data(name, ogb_root, device):
    if name in ('ogbn-products', 'ogbn-arxiv'):
        data = DglNodePropPredDataset(name, ogb_root)
        g, labels = data[0]
        if name == 'ogbn-arxiv':
            g = dgl.to_bidirected(g, copy_ndata=True)
            feat = g.ndata['feat']
            feat = (feat - feat.mean(dim=0)) / feat.std(dim=0)
            g.ndata['feat'] = feat
        g = g.to(device)
        labels = labels.squeeze(dim=1).to(device)
        split_idx = data.get_idx_split()
        train_idx = split_idx['train'].to(device)
        val_idx = split_idx['valid'].to(device)
        test_idx = split_idx['test'].to(device)
        return g, labels, data.num_classes, train_idx, val_idx, test_idx
    else:
        data = load_citation_dataset(name)
        g = data[0].to(device)
        train_idx = g.ndata['train_mask'].nonzero(as_tuple=True)[0]
        val_idx = g.ndata['val_mask'].nonzero(as_tuple=True)[0]
        test_idx = g.ndata['test_mask'].nonzero(as_tuple=True)[0]
        return g, g.ndata['label'], data.num_classes, train_idx, val_idx, test_idx


def train_base_model(base_model, feats, labels, train_idx, val_idx, test_idx, args):
    print(f'Base model {args.base_model}')
    optimizer = optim.Adam(base_model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        base_model.train()
        logits = base_model(feats)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx], labels[train_idx])
        val_acc = evaluate(base_model, feats, labels, val_idx)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, loss, train_acc, val_acc
        ))
    test_acc = evaluate(base_model, feats, labels, test_idx)
    print('Test Acc {:.4f}'.format(test_acc))


@torch.no_grad()
def evaluate(model, feats, labels, mask):
    model.eval()
    logits = model(feats)
    return accuracy(logits[mask], labels[mask])


def correct_and_smooth(base_model, g, feats, labels, train_idx, val_idx, test_idx, args):
    print('C&S')
    base_model.eval()
    base_pred = base_model(feats).softmax(dim=1)  # 注意要softmax

    cs = CorrectAndSmooth(
        args.num_correct_layers, args.correct_alpha, args.correct_norm,
        args.num_smooth_layers, args.smooth_alpha, args.smooth_norm, args.scale
    )
    mask = torch.cat([train_idx, val_idx])
    logits = cs(g, F.one_hot(labels).float(), base_pred, mask)
    test_acc = accuracy(logits[test_idx], labels[test_idx])
    print('Test Acc {:.4f}'.format(test_acc))


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    g, labels, num_classes, train_idx, val_idx, test_idx = \
        load_data(args.dataset, args.ogb_root, device)
    feats = g.ndata['feat']

    if args.base_model == 'Linear':
        base_model = nn.Linear(feats.shape[1], num_classes)
    else:
        base_model = MLP(feats.shape[1], args.num_hidden, num_classes, args.num_layers, args.dropout)
    base_model = base_model.to(device)
    train_base_model(base_model, feats, labels, train_idx, val_idx, test_idx, args)
    correct_and_smooth(base_model, g, feats, labels, train_idx, val_idx, test_idx, args)


def main():
    parser = argparse.ArgumentParser(description='Correct and Smooth')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument(
        '--dataset', choices=['cora', 'citeseer', 'pubmed', 'ogbn-products', 'ogbn-arxiv'],
        default='cora', help='dataset'
    )
    parser.add_argument('--ogb-root', type=str, help='root directory to OGB datasets')
    # Base model
    parser.add_argument('--base-model', choices=['Linear', 'MLP'], default='Linear', help='base model')
    parser.add_argument('--num-hidden', type=int, default=256, help='number of MLP hidden units')
    parser.add_argument('--num-layers', type=int, default=3, help='number of MLP layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='MLP dropout probability')
    parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    # C&S
    parser.add_argument(
        '--num-correct-layers', type=int, default=50,
        help='number of Correct propagation layers'
    )
    parser.add_argument('--correct-alpha', type=float, default=0.5, help='α of Correct')
    parser.add_argument(
        '--correct-norm', choices=['left', 'right', 'both'], default='both',
        help='normalization mode of Correct'
    )
    parser.add_argument(
        '--num-smooth-layers', type=int, default=50,
        help='number of Smooth propagation layers'
    )
    parser.add_argument('--smooth-alpha', type=float, default=0.5, help='α of Smooth')
    parser.add_argument(
        '--smooth-norm', choices=['left', 'right', 'both'], default='both',
        help='normalization mode of Smooth'
    )
    parser.add_argument('--scale', type=float, default=20, help='scaling factor')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
