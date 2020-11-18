import argparse

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from gnn.HAN.utils import set_random_seed, load_data


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])
    return loss, accuracy, micro_f1, macro_f1


def train(args):
    set_random_seed(args.seed)

    # If args.hetero is True, g would be a heterogeneous graph
    # Otherwise, it will be two homogeneous graphs
    dataset = 'ACMRaw' if args.hetero else 'ACM'
    data = load_data(dataset)
    g = data[0]
    ndata = g[0].ndata if dataset == 'ACM' else g.nodes['paper'].data
    features = ndata['feat']
    labels = ndata['label']
    train_mask = ndata['train_mask'].type(torch.bool)
    val_mask = ndata['val_mask'].type(torch.bool)
    test_mask = ndata['test_mask'].type(torch.bool)

    heads = [args.num_heads] * args.num_layers
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.hetero:
        from gnn.HAN.model_hetero import HAN
        model = HAN(
            meta_paths=[['pa', 'ap'], ['pf', 'fp']],
            in_size=features.shape[1],
            hidden_size=args.num_hidden,
            out_size=data.num_classes,
            num_heads=heads,
            dropout=args.dropout
        ).to(device)
        g = g.to(device)
    else:
        from gnn.HAN.model import HAN
        model = HAN(
            num_meta_paths=len(g),
            in_size=features.shape[1],
            hidden_size=args.num_hidden,
            out_size=data.num_classes,
            num_heads=heads,
            dropout=args.dropout
        ).to(device)
        g = [graph.to(device) for graph in g]

    loss_fcn = F.cross_entropy
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model, g, features, labels, val_mask, loss_fcn
        )
        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1,
            val_macro_f1))

    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model, g, features, labels, test_mask, loss_fcn
    )
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1
    ))


def main():
    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='number of attention heads for node-level attention')
    parser.add_argument('--num-layers', type=int, default=1, help='number of han layers')
    parser.add_argument('--num-hidden', type=int, default=8, help='number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()