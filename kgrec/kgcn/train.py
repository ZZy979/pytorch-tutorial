import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from gnn.utils import set_random_seed, get_device
from kgrec.kgcn.data import RatingKnowledgeGraphDataset
from kgrec.kgcn.dataloader import KGCNEdgeDataLoader
from kgrec.kgcn.model import KGCN


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data = RatingKnowledgeGraphDataset(args.dataset)
    user_item_graph = data.user_item_graph
    knowledge_graph = dgl.sampling.sample_neighbors(
        data.knowledge_graph, data.knowledge_graph.nodes(), args.neighbor_size, replace=True
    )

    train_eids, test_eids = train_test_split(
        torch.arange(user_item_graph.num_edges()), train_size=args.train_size,
        random_state=args.seed
    )
    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_hops)
    train_loader = KGCNEdgeDataLoader(
        user_item_graph, train_eids, sampler, knowledge_graph,
        device=device, batch_size=args.batch_size
    )
    test_loader = KGCNEdgeDataLoader(
        user_item_graph, test_eids, sampler, knowledge_graph,
        device=device, batch_size=args.batch_size
    )

    model = KGCN(args.num_hidden, args.neighbor_size, args.aggregator, args.num_hops, *data.get_num()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _, pair_graph, blocks in train_loader:
            scores = model(pair_graph, blocks)
            loss = F.binary_cross_entropy(scores, pair_graph.edata['label'])
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch {:d} | Train Loss {:.4f} | Train AUC {:.4f} | Train F1 {:.4f} | Test AUC {:.4f} | Test F1 {:.4f}'.format(
            epoch, sum(losses) / len(losses), *evaluate(model, train_loader), *evaluate(model, test_loader)
        ))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    auc_scores, f1_scores = [], []
    for _, pair_graph, blocks in loader:
        u, v = pair_graph.edges()
        scores = model(pair_graph, blocks).detach().cpu().numpy()
        labels = pair_graph.edata['label'].int().cpu().numpy()
        auc_scores.append(roc_auc_score(labels, scores))
        f1_scores.append(f1_score(labels, scores > 0.5))
    return sum(auc_scores) / len(auc_scores), sum(f1_scores) / len(f1_scores)


def main():
    parser = argparse.ArgumentParser(description='KGCN Link Prediction')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--dataset', choices=['movie', 'music'], default='music', help='dataset')
    parser.add_argument('--train-size', type=float, default=0.8, help='size of training dataset')
    parser.add_argument('--num-hidden', type=int, default=16, help='number of hidden units')
    parser.add_argument('--aggregator', choices=['sum', 'concat', 'neighbor'], help='aggregator')
    parser.add_argument('--num-hops', type=int, default=1, help='number of hops')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--neighbor-size', type=int, default=8, help='number of sampled neighbors')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
