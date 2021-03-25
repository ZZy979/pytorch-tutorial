"""SIGN: Scalable Inception Graph Neural Networks (SIGN)

* 论文链接：https://arxiv.org/pdf/2004.11198
* DGL实现：https://github.com/dmlc/dgl/tree/master/examples/pytorch/sign
"""
import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    """L层全连接网络"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.0):
        super().__init__()
        self.fc = nn.ModuleList()
        if num_layers == 1:
            self.fc.append(nn.Linear(in_dim, out_dim, bias=False))
        else:
            self.fc.append(nn.Linear(in_dim, hidden_dim, bias=False))
            for _ in range(num_layers - 2):
                self.fc.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.fc.append(nn.Linear(hidden_dim, out_dim, bias=False))
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: tensor(N, d_in)
        :return: tensor(N, d_out)
        """
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i < len(self.fc) - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_hops, num_layers, dropout=0.0):
        """SIGN模型

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_hops: int 跳数r
        :param num_layers: int 全连接网络层数
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.inception_ffs = nn.ModuleList([
            FeedForwardNet(in_dim, hidden_dim, hidden_dim, num_layers, dropout)  # (4)式中的Θ_i
            for _ in range(num_hops + 1)
        ])
        self.project = FeedForwardNet(
            (num_hops + 1) * hidden_dim, hidden_dim, out_dim, num_layers, dropout
        )  # (4)式中的Ω
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats):
        """
        :param feats: List[tensor(N, d_in)] 每一跳的邻居聚集特征，长度为r+1
        :return: tensor(N, d_out) 输出顶点特征
        """
        # (N, (r+1)*d_hid)
        h = torch.cat([ff(feat) for ff, feat in zip(self.inception_ffs, feats)], dim=-1)
        out = self.project(self.dropout(self.prelu(h)))  # (N, d_out)
        return out
