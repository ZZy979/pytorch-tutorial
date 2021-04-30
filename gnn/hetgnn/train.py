import argparse
import os
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.data.utils import load_graphs, load_info

from gnn.hetgnn.model import HetGNN
from gnn.hetgnn.utils import construct_neg_graph
from gnn.utils import set_random_seed, RatioNegativeSampler


def train(args):
    set_random_seed(args.seed)
    g = load_graphs(os.path.join(args.data_path, 'neighbor_graph.bin'))[0][0]
    feats = load_info(os.path.join(args.data_path, 'in_feats.pkl'))

    model = HetGNN(feats['author'].shape[-1], args.num_hidden, g.ntypes, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    neg_sampler = RatioNegativeSampler()
    for epoch in range(args.epochs):
        model.train()
        embeds = model(g, feats)
        score = model.calc_score(g, embeds)
        neg_g = construct_neg_graph(g, neg_sampler)
        neg_score = model.calc_score(neg_g, embeds)
        logits = torch.cat([score, neg_score])  # (2A*E,)
        labels = torch.cat([torch.ones(score.shape[0]), torch.zeros(neg_score.shape[0])])
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, loss.item()))
    with torch.no_grad():
        final_embeds = model(g, feats)
        with open(args.save_node_embed_path, 'wb') as f:
            pickle.dump(final_embeds, f)
        print('Final node embeddings saved to', args.save_node_embed_path)


def main():
    parser = argparse.ArgumentParser(description='HetGNN unsupervised training')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--num-hidden', type=int, default=128, help='number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('data_path', help='path to preprocessed data')
    parser.add_argument('save_node_embed_path', help='path to save learned final node embeddings')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
