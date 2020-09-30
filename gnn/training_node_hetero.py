"""在异构图上训练用于顶点分类/回归任务的GNN

参考：

* <https://docs.dgl.ai/guide/training-node.html>
* <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify.py>
"""
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import HeteroGraphConv, GraphConv


class RGCN(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats) for rel in rel_names
        }, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats) for rel in rel_names
        }, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)['user']
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main():
    # 异构图示例：https://docs.dgl.ai/guide/training.html#heterogeneous-graphs
    n_users = 1000
    n_items = 500
    n_follows = 3000
    n_clicks = 5000
    n_dislikes = 500
    n_hetero_features = 10
    n_user_classes = 5
    n_max_clicks = 10

    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)

    g = dgl.heterograph({
        ('user', 'follow', 'user'): (follow_src, follow_dst),
        ('user', 'followed-by', 'user'): (follow_dst, follow_src),
        ('user', 'click', 'item'): (click_src, click_dst),
        ('item', 'clicked-by', 'user'): (click_dst, click_src),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
        ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)
    })

    g.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
    g.nodes['item'].data['feature'] = torch.randn(n_items, n_hetero_features)
    g.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
    g.edges['click'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()

    user_feats = g.nodes['user'].data['feature']
    item_feats = g.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    labels = g.nodes['user'].data['label']

    idx = np.arange(n_users)
    np.random.shuffle(idx)
    n_train = int(n_users * 0.6)
    n_valid = int(n_users * 0.2)
    train_mask = torch.zeros(n_users, dtype=torch.bool)
    train_mask[idx[:n_train]] = True
    valid_mask = torch.zeros(n_users, dtype=torch.bool)
    valid_mask[idx[n_train:n_train + n_valid]] = True
    test_mask = torch.zeros(n_users, dtype=torch.bool)
    test_mask[idx[n_train + n_valid:]] = True

    model = RGCN(n_hetero_features, 20, n_user_classes, g.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(5):
        model.train()
        # forward propagation by using all nodes and extracting the user embeddings
        logits = model(g, node_features)['user']
        # compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # compute validation accuracy
        acc = evaluate(model, g, node_features, labels, valid_mask)
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('Epoch {:d} | Loss {:.4f} | Accuracy {:.2%}'.format(epoch + 1, loss.item(), acc))
    acc = evaluate(model, g, node_features, labels, test_mask)
    print('Test accuracy {:.4f}'.format(acc))


if __name__ == '__main__':
    main()
