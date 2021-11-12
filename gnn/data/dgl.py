import dgl
import numpy as np
import torch
from dgl.data import DGLDataset


class UserItemDataset(DGLDataset):

    def __init__(self, n_users=1000, n_items=500, n_follows=3000, n_clicks=5000, n_dislikes=500,
                 n_features=10, n_user_classes=5, n_max_clicks=10):
        """“用户-物品”异构图

        https://docs.dgl.ai/en/latest/guide/training.html#heterogeneous-graphs

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
        user

        * feat - 维数为n_features
        * label - 范围[0, n_user_classes - 1]
        * train_mask

        item

        * feat - 维数为n_features

        边特征
        =====
        click

        * label - 范围[1, n_max_clicks]
        * train_mask

        :param n_users: “用户”顶点个数
        :param n_items: “物品”顶点个数
        :param n_follows: “用户-关注->用户”边条数
        :param n_clicks: “用户-点击->物品”边条数
        :param n_dislikes: “用户-不喜欢->物品”边条数
        :param n_features: “用户”和“物品”顶点特征维数
        :param n_user_classes: “用户”顶点标签类别数
        :param n_max_clicks: “用户-点击->物品”边标签类别数
        """
        super().__init__('user-item')
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
        g.nodes['user'].data['feat'] = torch.randn(n_users, n_features)
        g.nodes['item'].data['feat'] = torch.randn(n_items, n_features)
        g.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
        g.edges['click'].data['label'] = torch.randint(1, n_max_clicks + 1, (n_clicks,))
        # randomly generate training masks on user nodes and click edges
        g.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
        g.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)
        self.g = g

    def process(self):
        pass

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.g

    def __len__(self):
        return 1


class RandomGraphDataset(DGLDataset):

    def __init__(self, num_nodes, num_edges, num_feats):
        """随机图数据集

        :param num_nodes: int 顶点数
        :param num_edges: int 边数
        :param num_feats: int 顶点特征维数
        """
        super().__init__('random-graph')
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        # make it symmetric
        g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), num_nodes=num_nodes)
        # synthetic node features
        g.ndata['feat'] = torch.randn(num_nodes, num_feats)
        self.g = g

    def process(self):
        pass

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.g

    def __len__(self):
        return 1
