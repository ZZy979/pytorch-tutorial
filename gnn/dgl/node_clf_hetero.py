"""在异构图上训练用于顶点分类/回归任务的GNN

https://docs.dgl.ai/en/latest/guide/training-node.html
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


def build_user_item_graph(
        n_users=1000, n_items=500, n_follows=3000, n_clicks=5000, n_dislikes=500,
        n_features=10, n_user_classes=5, n_max_clicks=10
):
    """构造一个“用户-物品”异构图。

    顶点类型
    =====
    * user - 用户
    * item - 物品

    边类型
    =====
    * follow - 用户-关注->用户
    * follow-by - 用户-被关注->用户
    * click - 用户-点击->物品
    * click-by - 物品-被点击->用户
    * dislike - 用户-不喜欢->物品
    * dislike-by - 物品-被不喜欢->用户

    顶点特征
    =====
    * user
        * "feature" - 维数为n_features
        * "label" - 范围[0, n_user_classes - 1]
        * "train_mask"
    * item
        * "feature" - 维数为n_features

    边特征
    =====
    * click
        * "label" - 范围[1, n_max_clicks]
        * "train_mask"

    :param n_users: “用户”顶点个数
    :param n_items: “物品”顶点个数
    :param n_follows: “用户-关注->用户”边条数
    :param n_clicks: “用户-点击->物品”边条数
    :param n_dislikes: “用户-不喜欢->物品”边条数
    :param n_features: “用户”和“物品”顶点特征维数
    :param n_user_classes: “用户”顶点标签类别数
    :param n_max_clicks: “用户-点击->物品”边标签类别数
    """
    # 异构图示例：https://docs.dgl.ai/guide/training.html#heterogeneous-graphs
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
    }, num_nodes_dict={'user': n_users, 'item': n_items})
    g.nodes['user'].data['feature'] = torch.randn(n_users, n_features)
    g.nodes['item'].data['feature'] = torch.randn(n_items, n_features)
    g.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
    g.edges['click'].data['label'] = torch.randint(1, n_max_clicks + 1, (n_clicks,))
    # randomly generate training masks on user nodes and click edges
    g.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
    g.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)
    return g


def main():
    g = build_user_item_graph()
    user_feats = g.nodes['user'].data['feature']
    item_feats = g.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    labels = g.nodes['user'].data['label']
    train_mask = g.nodes['user'].data['train_mask']

    model = RGCN(user_feats.shape[1], 20, labels.max().item() + 1, g.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(5):
        model.train()
        # forward propagation by using all nodes and extracting the user embeddings
        logits = model(g, node_features)['user']
        # compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # compute validation accuracy, omitted in this example
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == '__main__':
    main()
