import argparse
import pickle

import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

from gnn.data import AMiner2Dataset


def load_data(node_embed_path):
    data = AMiner2Dataset()
    g = data[0]
    idx = torch.nonzero(g.nodes['author'].data['label'] != -1, as_tuple=True)[0]
    labels = g.nodes['author'].data['label'][idx].numpy()

    with open(node_embed_path, 'rb') as f:
        node_embeds = pickle.load(f)
    return node_embeds['author'][idx].numpy(), labels, data.num_classes


def node_clf(args):
    embeds, labels, _ = load_data(args.node_embed_path)
    for train_size in (0.1, 0.3):
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
    embeds, labels, num_classes = load_data(args.node_embed_path)
    pred = KMeans(num_classes, random_state=args.seed).fit_predict(embeds)
    nmi = normalized_mutual_info_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    print('NMI {:.4f} | ARI {:.4f}'.format(nmi, ari))


def main():
    parser = argparse.ArgumentParser(description='HetGNN node classification or clustering')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--task', choices=['clf', 'cluster'], default='clf', help='training task')
    parser.add_argument('node_embed_path', help='path to saved node embeddings')
    args = parser.parse_args()
    if args.task == 'clf':
        node_clf(args)
    else:
        node_cluster(args)


if __name__ == '__main__':
    main()
