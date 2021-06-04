import argparse
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from gensim.models import Word2Vec
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from tqdm import tqdm

from gnn.rhgnn.model import RHGNN
from gnn.utils import set_random_seed, get_device, add_reverse_edges


def load_data(path, device):
    data = DglNodePropPredDataset('ogbn-mag', path)
    g, labels = data[0]
    g = add_reverse_edges(g)
    labels = labels['paper'].to(device)
    split_idx = data.get_idx_split()
    train_idx = split_idx['train']['paper'].to(device)
    val_idx = split_idx['valid']['paper'].to(device)
    test_idx = split_idx['test']['paper'].to(device)
    return g, labels, data.num_classes, train_idx, val_idx, test_idx, Evaluator(data.name)


def load_pretrained_node_embed(g, path):
    model = Word2Vec.load(path)
    paper_embed = torch.from_numpy(model.wv[[f'paper_{i}' for i in range(g.num_nodes('paper'))]])
    g.nodes['paper'].data['feat'] = torch.cat([g.nodes['paper'].data['feat'], paper_embed], dim=1)
    for ntype in ('author', 'field_of_study', 'institution'):
        g.nodes[ntype].data['feat'] = torch.from_numpy(
            model.wv[[f'{ntype}_{i}' for i in range(g.num_nodes(ntype))]]
        )


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    g, labels, num_classes, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.ogb_path, device)
    load_pretrained_node_embed(g, args.node_embed_path)

    sampler = MultiLayerNeighborSampler(
        list(range(args.neighbor_size, args.neighbor_size + args.num_layers))
    )
    train_loader = NodeDataLoader(g, {'paper': train_idx}, sampler, batch_size=args.batch_size)
    val_loader = NodeDataLoader(g, {'paper': val_idx}, sampler, batch_size=args.batch_size)
    test_loader = NodeDataLoader(g, {'paper': test_idx}, sampler, batch_size=args.batch_size)

    model = RHGNN(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, num_classes, args.num_rel_hidden, args.num_rel_hidden, args.num_heads,
        g.ntypes, g.canonical_etypes, 'paper', args.num_layers, args.dropout, residual=args.residual
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=args.lr / 100
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        logits, train_labels, losses = [], [], []
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
            blocks = [b.to(device) for b in blocks]
            batch_labels = labels[output_nodes['paper']]
            batch_logits = model(blocks, blocks[0].srcdata['feat'])
            loss = F.cross_entropy(batch_logits, batch_labels.squeeze(dim=1))

            logits.append(batch_logits.detach().cpu())
            train_labels.append(batch_labels.detach().cpu())
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()

        train_acc = accuracy(torch.cat(logits, dim=0), torch.cat(train_labels, dim=0), evaluator)
        val_acc = evaluate(val_loader, device, model, labels, evaluator)
        test_acc = evaluate(test_loader, device, model, labels, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f}'.format(
            epoch, torch.tensor(losses).mean().item(), train_acc, val_acc, test_acc
        ))
    # embed = model.inference(g, g.ndata['feat'], device, args.batch_size)
    # test_acc = accuracy(embed[test_idx], labels[test_idx], evaluator)
    test_acc = evaluate(test_loader, device, model, labels, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


@torch.no_grad()
def accuracy(logits, labels, evaluator):
    predict = logits.argmax(dim=1, keepdim=True)
    return evaluator.eval({'y_true': labels, 'y_pred': predict})['acc']


@torch.no_grad()
def evaluate(loader, device, model, labels, evaluator):
    model.eval()
    logits, eval_labels = [], []
    for input_nodes, output_nodes, blocks in loader:
        blocks = [b.to(device) for b in blocks]
        batch_labels = labels[output_nodes['paper']]
        batch_logits = model(blocks, blocks[0].srcdata['feat'])

        logits.append(batch_logits.detach().cpu())
        eval_labels.append(batch_labels.detach().cpu())
    return accuracy(torch.cat(logits, dim=0), torch.cat(eval_labels, dim=0), evaluator)


def main():
    parser = argparse.ArgumentParser(description='R-HGNN ogbn-mag')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument(
        '--num-rel-hidden', type=int, default=8, help='number of relation hidden units'
    )
    parser.add_argument('--num-heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    parser.add_argument(
        '--no-residual', action='store_false', help='no residual connection', dest='residual'
    )
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='batch size')
    parser.add_argument('--neighbor-size', type=int, default=10, help='number of sampled neighbors')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('ogb_path', help='path to OGB datasets')
    parser.add_argument('node_embed_path', help='path to pretrained node embeddings')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
