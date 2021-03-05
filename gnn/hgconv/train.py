import argparse
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim

from gnn.data import ACMDataset, IMDbDataset
from gnn.hgconv.model import HGConv
from gnn.utils import set_random_seed, micro_macro_f1_score, nmi_ari_score

DATASET = {
    'acm': ACMDataset,
    'imdb': IMDbDataset
}


def train(args):
    set_random_seed(args.seed)
    data = DATASET[args.dataset]()
    g = data[0]
    predict_ntype = data.predict_ntype
    features = {ntype: g.nodes[ntype].data['feat'] for ntype in g.ntypes}
    labels = g.nodes[predict_ntype].data['label']
    train_mask = g.nodes[predict_ntype].data['train_mask']
    val_mask = g.nodes[predict_ntype].data['val_mask']
    test_mask = g.nodes[predict_ntype].data['test_mask']

    model = HGConv(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, data.num_classes, args.num_heads, g, predict_ntype,
        args.num_layers, args.dropout, args.residual
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    score = micro_macro_f1_score if args.task == 'clf' else nmi_ari_score
    if args.task == 'clf':
        metrics = 'Epoch {:d} | Train Loss {:.4f} | Train Micro-F1 {:.4f} | Train Macro-F1 {:.4f}' \
                  ' | Val Micro-F1 {:.4f} | Val Macro-F1 {:.4f}'
    else:
        metrics = 'Epoch {:d} | Train Loss {:.4f} | Train NMI {:.4f} | Train ARI {:.4f}' \
                  ' | Val NMI {:.4f} | Val ARI {:.4f}'
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_metrics = score(logits[train_mask], labels[train_mask])
        val_metrics = evaluate(model, g, features, labels, val_mask, score)
        print(metrics.format(epoch, loss.item(), *train_metrics, *val_metrics))

    test_metrics = evaluate(model, g, features, labels, test_mask, score)
    if args.task == 'clf':
        print('Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'.format(*test_metrics))
    else:
        print('Test NMI {:.4f} | Test ARI {:.4f}'.format(*test_metrics))


def evaluate(model, g, features, labels, mask, score):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    return score(logits[mask], labels[mask])


def main():
    parser = argparse.ArgumentParser(description='HGConv Node Classification or Clustering')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', choices=['acm', 'imdb'], default='acm', help='dataset')
    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--num-heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    parser.add_argument(
        '--no-residual', action='store_false', help='no residual connection', dest='residual'
    )
    parser.add_argument('--task', choices=['clf', 'cluster'], default='clf', help='training task')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
