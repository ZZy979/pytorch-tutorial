import argparse
import random

import torch
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from gnn.data import AMinerCSDataset


def load_data(ntype, model_path):
    data = AMinerCSDataset()
    g = data[0]
    idx = torch.nonzero(g.nodes[ntype].data['label'] != -1, as_tuple=True)[0]
    labels = g.nodes[ntype].data['label'][idx].numpy()

    model = Word2Vec.load(model_path)
    names = data.author_names if ntype == 'author' else data.conf_names
    embeds = model.wv[[names[i] for i in idx]]
    return embeds, labels, data.num_classes


def node_clf(args):
    embeds, labels, _ = load_data(args.ntype, args.model_path)
    for train_size in [0.05] + [0.1 * x for x in range(1, 10)]:
        X_train, X_test, y_train, y_test = train_test_split(
            embeds, labels, train_size=train_size, random_state=args.seed
        )
        clf = LogisticRegression(random_state=args.seed, max_iter=500)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print('Train size {:.0%} | Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'.format(
            train_size, micro_f1, macro_f1
        ))


def node_cluster(args):
    embeds, labels, num_classes = load_data(args.ntype, args.model_path)
    random.seed(args.seed)
    seeds = [random.randint(0, 0x7fffffff) for _ in range(10)]
    nmis = []
    for seed in seeds:
        pred = KMeans(num_classes, random_state=seed).fit_predict(embeds)
        nmi = normalized_mutual_info_score(labels, pred)
        print('Seed {:d} | NMI {:.4f}'.format(seed, nmi))
        nmis.append(nmi)
    print('Average NMI {:.4f}'.format(sum(nmis) / len(nmis)))


def main():
    parser = argparse.ArgumentParser(description='metapath2vec Node Classification or Clustering')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--ntype', choices=['author', 'conf'], default='author', help='node type')
    parser.add_argument('--task', choices=['clf', 'cluster'], default='clf', help='training task')
    parser.add_argument('model_path', help='path to word2vec model (node embeddings)')
    args = parser.parse_args()
    if args.task == 'clf':
        node_clf(args)
    else:
        node_cluster(args)


if __name__ == '__main__':
    main()
