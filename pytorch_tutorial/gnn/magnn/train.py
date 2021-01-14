import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeCollator
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from pytorch_tutorial.gnn.data import DBLPFourAreaDataset
from pytorch_tutorial.gnn.magnn.model import MAGNNMinibatch
from pytorch_tutorial.gnn.utils import set_random_seed, metapath_based_graph, to_ntype_list, \
    micro_macro_f1_score


def train(args):
    set_random_seed(args.seed)
    data = DBLPFourAreaDataset()
    g = data[0]
    metapaths = data.metapaths
    predict_ntype = data.predict_ntype
    generate_one_hot_id(g)
    features = g.ndata['feat']  # Dict[str, tensor(N_i, d_i)]
    labels = g.nodes[predict_ntype].data['label']
    train_idx = g.nodes[predict_ntype].data['train_mask'].nonzero(as_tuple=True)[0]

    # 基于元路径的邻居组成的同构图
    print('正在生成基于元路径的图...')
    mgs = [metapath_based_graph(g, metapath) for metapath in metapaths]
    mgs[0].ndata['feat'] = features[predict_ntype]
    sampler = MultiLayerFullNeighborSampler(1)
    collators = [NodeCollator(mg, None, sampler) for mg in mgs]
    train_dataloader = DataLoader(train_idx, batch_size=args.batch_size)

    metapaths_ntype = [to_ntype_list(g, metapath) for metapath in metapaths]
    model = MAGNNMinibatch(
        predict_ntype, metapaths_ntype,
        {ntype: feat.shape[1] for ntype, feat in features.items()},
        args.num_hidden, data.num_classes, args.num_heads, args.encoder, args.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        train_logits = torch.zeros((g.num_nodes(predict_ntype), data.num_classes))
        for batch in tqdm(train_dataloader):
            gs = [collator.collate(batch)[2][0] for collator in collators]
            train_logits[batch] = logits = model(gs, features)
            loss = F.cross_entropy(logits, labels[batch])
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = micro_macro_f1_score(train_logits[train_idx], labels[train_idx])
        print('Epoch {:d} | Train Loss {:.4f} | Train Micro-F1 {:.4f} | Train Macro-F1 {:.4f}'.format(
            epoch, torch.tensor(losses).mean().item(), *train_metrics
        ))


def generate_one_hot_id(g):
    for ntype in g.ntypes:
        if 'feat' not in g.nodes[ntype].data:
            g.nodes[ntype].data['feat'] = torch.eye(g.num_nodes(ntype))


def main():
    parser = argparse.ArgumentParser(description='MAGNN Node Classification (minibatch)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num-hidden', type=int, default=64, help='dimension of hidden state')
    parser.add_argument('--num-heads', type=int, default=8, help='number of attention heads')
    parser.add_argument(
        '--encoder', choices=['linear'], default='linear', help='metapath instance encoder'
    )
    parser.add_argument('--dropout', type=float, default=0.5, help='feature and attention dropout')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
