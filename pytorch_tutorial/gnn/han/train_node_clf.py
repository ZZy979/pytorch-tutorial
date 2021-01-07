import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

from pytorch_tutorial.gnn.data import *
from pytorch_tutorial.gnn.han.model import NodeClassification
from pytorch_tutorial.gnn.utils import set_random_seed

DATASET = {
    'acm': ACM3025Dataset,
    'dblp': DBLP4057Dataset
}

HETERO_DATASET = {
    'acm': ACMDataset,
    'dblp': DBLPFourAreaDataset
}


def train(args):
    set_random_seed(args.seed)
    if args.hetero:
        data = HETERO_DATASET[args.dataset]()
        g = data[0]
        gs = [dgl.metapath_reachable_graph(g, metapath) for metapath in data.metapaths]
        for i in range(len(gs)):
            gs[i] = dgl.add_self_loop(dgl.remove_self_loop(gs[i]))
        ntype = data.predict_ntype
        num_classes = data.num_classes
        features = g.nodes[ntype].data['feat']
        labels = g.nodes[ntype].data['label']
        train_mask = g.nodes[ntype].data['train_mask']
        val_mask = g.nodes[ntype].data['val_mask']
        test_mask = g.nodes[ntype].data['test_mask']
    else:
        data = DATASET[args.dataset]()
        gs = data[0]
        num_classes = data.num_classes
        features = gs[0].ndata['feat']
        labels = gs[0].ndata['label']
        train_mask = gs[0].ndata['train_mask']
        val_mask = gs[0].ndata['val_mask']
        test_mask = gs[0].ndata['test_mask']

    model = NodeClassification(
        len(gs), features.shape[1], args.num_hidden, num_classes, args.num_heads, args.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits = model(gs, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_acc, val_micro_f1, val_macro_f1 = score(logits[val_mask], labels[val_mask])
        print(
            'Epoch {:d} | Train Loss {:.4f} | Train Micro-F1 {:.4f} | Train Macro-F1 {:.4f} | '
            'Val Micro-F1 {:.4f} | Val Macro-F1 {:.4f}'.format(
                epoch, loss.item(), train_micro_f1, train_macro_f1, val_micro_f1, val_macro_f1
            )
        )

    test_acc, test_micro_f1, test_macro_f1 = evaluate(model, gs, features, labels, test_mask)
    print('Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'.format(test_micro_f1, test_macro_f1))


def score(logits, labels):
    prediction = torch.argmax(logits, dim=1).long().numpy()
    labels = labels.numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1


def evaluate(model, gs, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(gs, features)
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])
    return accuracy, micro_f1, macro_f1


def main():
    parser = argparse.ArgumentParser('HAN Node Classification')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', choices=['acm', 'dblp'], default='acm', help='dataset')
    parser.add_argument(
        '--hetero', action='store_true', help='Use heterogeneous graph dataset'
    )
    parser.add_argument('--num-hidden', type=int, default=8, help='number of hidden units')
    parser.add_argument(
        '--num-heads', type=int, default=8, help='number of attention heads in node-level attention'
    )
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
