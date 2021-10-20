import argparse
import random

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score, adjusted_rand_score

from gnn.data import ACMHeCoDataset, DBLPHeCoDataset, FreebaseHeCoDataset, AMinerHeCoDataset
from gnn.heco.model import HeCo
from gnn.utils import set_random_seed, get_device, accuracy, micro_macro_f1_score

DATASET = {
    'acm': {
        'class': ACMHeCoDataset,
        'relations': [('author', 'ap', 'paper'), ('subject', 'sp', 'paper')],
        'neighbor_sizes': [7, 1],
        'pos_threshold': 5
    },
    'dblp': {
        'class': DBLPHeCoDataset,
        'relations': [('paper', 'pa', 'author')],
        'neighbor_sizes': [6],
        'pos_threshold': 1000
    },
    'freebase': {
        'class': FreebaseHeCoDataset,
        'relations': [('author', 'am', 'movie'), ('director', 'dm', 'movie'), ('writer', 'wm', 'movie')],
        'neighbor_sizes': [18, 1, 2],
        'pos_threshold': 80
    },
    'aminer': {
        'class': AMinerHeCoDataset,
        'relations': [('author', 'ap', 'paper'), ('reference', 'rp', 'paper')],
        'neighbor_sizes': [3, 8],
        'pos_threshold': 15
    }
}


def load_data(name, device):
    dataset = DATASET[name]
    data = dataset['class']()
    g = data[0]
    predict_ntype = data.predict_ntype
    relations = dataset['relations']  # 网络结构视图编码器考虑的邻居类型
    neighbor_sizes = dataset['neighbor_sizes']
    pos_threshold = dataset['pos_threshold']

    pos = torch.zeros((g.num_nodes(predict_ntype), g.num_nodes(predict_ntype)), dtype=torch.int, device=device)
    pos[data.pos] = 1

    for ntype in g.ntypes:
        if ntype != predict_ntype or 'feat' not in g.nodes[ntype].data:
            g.nodes[ntype].data['feat'] = torch.eye(g.num_nodes(ntype))
    feats = [g.nodes[predict_ntype].data['feat']] + [g.nodes[r[0]].data['feat'] for r in relations]
    feats = [feat.to(device) for feat in feats]
    labels = g.nodes[predict_ntype].data['label']
    train_mask = g.nodes[predict_ntype].data['train_mask']
    val_mask = g.nodes[predict_ntype].data['val_mask']
    test_mask = g.nodes[predict_ntype].data['test_mask']
    return data, g, feats, labels, predict_ntype, relations, neighbor_sizes, \
        pos, pos_threshold, train_mask, val_mask, test_mask


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, feats, labels, predict_ntype, relations, neighbor_sizes, \
        pos, pos_threshold, train_mask, val_mask, test_mask = load_data(args.dataset, device)
    bgs = [g[rel] for rel in relations]  # 邻居-目标顶点二分图
    mgs = [
        dgl.add_self_loop(dgl.remove_self_loop(dgl.metapath_reachable_graph(g, mp))).to(device)
        for mp in data.metapaths
    ]  # 基于元路径的邻居同构图

    model = HeCo(
        [feat.shape[1] for feat in feats], args.num_hidden, args.feat_drop, args.attn_drop,
        neighbor_sizes, len(data.metapaths), args.tau, args.lambda_
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        loss = model(bgs, mgs, feats, pos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {:d} | Train Loss {:.4f}'.format(epoch, loss.item()))
    evaluate(model, mgs, feats[0], labels, data.num_classes, train_mask, test_mask, args.seed)


def evaluate(model, mgs, feat, labels, num_classes, train_mask, test_mask, seed):
    model.eval()
    embeds = model.get_embeds(mgs, feat).cpu()

    micro_f1, macro_f1, auc = node_clf(embeds, labels, num_classes, train_mask, test_mask)
    print('Micro-F1 {:.4f} | Macro-F1 {:.4f} | AUC {:.4f}'.format(micro_f1, macro_f1, auc))

    nmi, ari = node_cluster(embeds.numpy(), labels.numpy(), num_classes)
    print('NMI {:.4f} | ARI {:.4f}'.format(nmi, ari))


def node_clf(embeds, labels, num_classes, train_mask, test_mask):
    clf = nn.Linear(embeds.shape[1], num_classes)
    optimizer = optim.Adam(clf.parameters(), lr=0.05)
    best_acc, best_logits = 0, None
    for epoch in range(200):
        clf.train()
        logits = clf(embeds[train_mask])
        loss = F.cross_entropy(logits, labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            clf.eval()
            test_logits = clf(embeds[test_mask])
            if accuracy(test_logits, labels[test_mask]) > best_acc:
                best_logits = test_logits
    micro_f1, macro_f1 = micro_macro_f1_score(best_logits, labels[test_mask])
    y_score = best_logits.softmax(dim=1).numpy()
    auc = roc_auc_score(labels[test_mask].numpy(), y_score, multi_class='ovr')
    return micro_f1, macro_f1, auc


def node_cluster(embeds, labels, num_classes):
    seeds = [random.randint(0, 0x7fffffff) for _ in range(10)]
    nmi, ari = [], []
    for seed in seeds:
        pred = KMeans(num_classes, random_state=seed).fit_predict(embeds)
        nmi.append(normalized_mutual_info_score(labels, pred))
        ari.append(adjusted_rand_score(labels, pred))
    return sum(nmi) / len(nmi), sum(ari) / len(ari)


def main():
    parser = argparse.ArgumentParser(description='HeCo')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument(
        '--dataset', choices=['acm', 'dblp', 'freebase', 'aminer'], default='acm', help='dataset'
    )
    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--feat-drop', type=float, default=0.3, help='feature dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='attention dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='temperature parameter')
    parser.add_argument(
        '--lambda', type=float, default=0.5, dest='lambda_',
        help='balance coefficient of contrastive loss'
    )
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0008, help='learning rate')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
