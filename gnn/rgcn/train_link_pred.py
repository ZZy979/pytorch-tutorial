import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading.negative_sampler import Uniform

from gnn.rgcn.model import LinkPrediction
from gnn.utils import load_kg_dataset


def train(args):
    data = load_kg_dataset(args.dataset)
    g = data[0]
    train_idx = g.edata['train_mask'].nonzero(as_tuple=False).squeeze()
    val_idx = g.edata['val_mask'].nonzero(as_tuple=False).squeeze()
    test_idx = g.edata['test_mask'].nonzero(as_tuple=False).squeeze()

    train_g = dgl.edge_subgraph(g, train_idx, preserve_nodes=True)
    train_triplets = g.find_edges(train_idx) + (train_g.edata['etype'],)
    model = LinkPrediction(
        data.num_nodes, args.num_hidden, data.num_rels * 2, args.num_layers,
        args.regularizer, args.num_bases, args.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    neg_sampler = Uniform(args.num_neg_samples)
    labels = torch.cat([torch.ones(train_g.num_edges()), torch.zeros(train_g.num_edges() * args.num_neg_samples)] )
    for epoch in range(args.epochs):
        model.train()
        embed = model(train_g, train_g.edata['etype'])

        neg_triplets = neg_sampler(train_g, torch.arange(train_g.num_edges())) \
            + (train_g.edata['etype'].repeat_interleave(args.num_neg_samples),)
        pos_score = model.calc_score(embed, train_triplets)
        neg_score = model.calc_score(embed, neg_triplets)
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos_score, neg_score]), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO 计算MRR
        # FB-15k和FB15k-237反向传播很慢？
        print('Epoch {:04d} | Loss {:.4f}'.format(epoch, loss.item()))


def main():
    parser = argparse.ArgumentParser(description='R-GCN Link Prediction')
    parser.add_argument(
        '--dataset', choices=['wn18', 'FB15k', 'FB15k-237'], default='wn18', help='dataset'
    )
    parser.add_argument('--num-hidden', type=int, default=200, help='number of hidden units')
    parser.add_argument('--num-layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument(
        '--regularizer', choices=['basis', 'bdd'], default='basis', help='weight regularizer'
    )
    parser.add_argument('--num-bases', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--num-neg-samples', type=int, default=1, help='number of negative samples')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
