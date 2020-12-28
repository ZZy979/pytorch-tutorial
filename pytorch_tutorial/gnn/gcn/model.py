"""Graph Convolutional Network (GCN)

* 论文链接：https://arxiv.org/abs/1609.02907
* DGL教程：https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
* DGL实现：https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn
"""
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class GCN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=F.relu)
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = self.conv2(g, h)
        return h
