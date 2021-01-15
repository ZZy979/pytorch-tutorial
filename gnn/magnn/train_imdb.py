import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from gnn.data import IMDbDataset
from gnn.magnn.encoder import ENCODERS
from gnn.magnn.model import MAGNNMultiLayer
from gnn.utils import set_random_seed, metapath_based_graph, to_ntype_list, micro_macro_f1_score

METAPATHS = {
    'movie': [['ma', 'am'], ['md', 'dm']],
    'director': [['dm', 'md'], ['dm', 'ma', 'am', 'md']],
    'actor': [['am', 'ma'], ['am', 'md', 'dm', 'ma']]
}


def train(args):
    set_random_seed(args.seed)
    data = IMDbDataset()
    g = data[0]
    predict_ntype = data.predict_ntype
    features = g.ndata['feat']  # Dict[str, tensor(N_i, d_i)]
    labels = g.nodes[predict_ntype].data['label']
    train_mask = g.nodes[predict_ntype].data['train_mask']
    val_mask = g.nodes[predict_ntype].data['val_mask']
    test_mask = g.nodes[predict_ntype].data['test_mask']

    print('正在生成基于元路径的图...')
    mgs = {
        ntype: [metapath_based_graph(g, metapath) for metapath in METAPATHS[ntype]]
        for ntype in METAPATHS
    }
    for ntype in mgs:
        mgs[ntype][0].ndata['feat'] = g.nodes[ntype].data['feat']
    metapaths_ntype = {
        ntype: [to_ntype_list(g, metapath) for metapath in METAPATHS[ntype]]
        for ntype in METAPATHS
    }

    model = MAGNNMultiLayer(
        args.num_layers, metapaths_ntype,
        {ntype: feat.shape[1] for ntype, feat in features.items()},
        args.num_hidden, data.num_classes, args.num_heads, args.encoder, args.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits = model(mgs, features)[predict_ntype]
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_metrics = micro_macro_f1_score(logits[train_mask], labels[train_mask])
        print('Epoch {:d} | Train Loss {:.4f} | Train Micro-F1 {:.4f} | Train Macro-F1 {:.4f}'.format(
            epoch, loss.item(), *train_metrics
        ))
        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate(model, mgs, features, predict_ntype, labels, val_mask)
            print('Val Micro-F1 {:.4f} | Val Macro-F1 {:.4f}'.format(*val_metrics))

    test_metrics = evaluate(model, mgs, features, predict_ntype, labels, test_mask)
    print('Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'.format(*test_metrics))


def evaluate(model, gs, features, predict_ntype, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(gs, features)[predict_ntype]
    return micro_macro_f1_score(logits[mask], labels[mask])


def main():
    parser = argparse.ArgumentParser(description='MAGNN Node Classification (multi-layer)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num-layers', type=int, default=2, help='number of MAGNN layers')
    parser.add_argument('--num-hidden', type=int, default=64, help='dimension of hidden state')
    parser.add_argument('--num-heads', type=int, default=8, help='number of attention heads')
    parser.add_argument(
        '--encoder', choices=list(ENCODERS.keys()), default='linear',
        help='metapath instance encoder'
    )
    parser.add_argument('--dropout', type=float, default=0.5, help='feature and attention dropout')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
