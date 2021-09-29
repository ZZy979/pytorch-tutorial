"""Graph Convolutional Network (GCN)

论文链接：https://arxiv.org/abs/1609.02907
"""
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class GCN(nn.Module):
    """两层GCN模型"""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=F.relu)
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, g, x):
        """
        :param g: DGLGraph
        :param x: tensor(N, d_in) 输入顶点特征，N为g的顶点数
        :return: tensor(N, d_out) 输出顶点特征
        """
        h = self.conv1(g, x)  # (N, d_hid)
        h = self.conv2(g, h)  # (N, d_out)
        return h
