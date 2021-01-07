"""Heterogeneous Graph Attention Network (HAN)

* 论文链接：https://arxiv.org/abs/1903.07293
* 官方代码：https://github.com/Jhy1993/HAN
* DGL实现：https://github.com/dmlc/dgl/tree/master/examples/pytorch/han
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv


class SemanticAttention(nn.Module):

    def __init__(self, in_dim, hidden_dim=128):
        """语义层次的注意力，将顶点基于不同元路径的嵌入组合为最终嵌入

        :param in_dim: 输入特征维数d_in，对应顶点层次注意力的输出维数
        :param hidden_dim: 语义层次隐含特征维数d_hid
        """
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # 论文公式(7)中的W和b
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)  # 论文公式(7)中的q
        )

    def forward(self, z):
        """
        :param z: tensor(N, M, d_in) 顶点基于不同元路径的嵌入，N为顶点数，M为元路径个数
        :return: tensor(N, d_in) 顶点的最终嵌入
        """
        w = self.project(z).mean(dim=0)  # (N, M, d_hid) -> (N, M, 1) -> (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        z = (beta * z).sum(dim=1)  # (N, d_in)
        return z


class HANLayer(nn.Module):

    def __init__(self, num_metapaths, in_dim, out_dim, num_heads, dropout):
        """HAN层

        :param num_metapaths: int 元路径个数
        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param dropout: float Dropout概率
        """
        super().__init__()
        # 顶点层次的注意力，每个GAT层对应一个元路径
        self.gats = nn.ModuleList([
            GATConv(in_dim, out_dim, num_heads, dropout, dropout, activation=F.elu)
            for _ in range(num_metapaths)
        ])
        # 语义层次的注意力
        self.semantic_attention = SemanticAttention(in_dim=num_heads * out_dim)

    def forward(self, gs, h):
        """
        :param gs: List[DGLGraph] 基于元路径的邻居组成的同构图
        :param h: tensor(N, d_in) 输入顶点特征
        :return: tensor(N, K*d_out) 输出顶点特征
        """
        zp = [gat(g, h).flatten(start_dim=1) for gat, g in zip(self.gats, gs)]  # 基于元路径的嵌入
        zp = torch.stack(zp, dim=1)  # (N, M, K*d_out)
        z = self.semantic_attention(zp)  # (N, K*d_out)
        return z


class NodeClassification(nn.Module):

    def __init__(self, num_metapaths, in_dim, hidden_dim, out_dim, num_heads, dropout):
        """HAN顶点分类模型

        :param num_metapaths: int 元路径个数
        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param dropout: float Dropout概率
        """
        super().__init__()
        self.han = HANLayer(num_metapaths, in_dim, hidden_dim, num_heads, dropout)
        self.predict = nn.Linear(num_heads * hidden_dim, out_dim)

    def forward(self, gs, h):
        """
        :param gs: List[DGLGraph] 基于元路径的邻居组成的同构图
        :param h: tensor(N, d_in) 输入顶点特征
        :return: tensor(N, d_out) 输出顶点嵌入
        """
        h = self.han(gs, h)  # (N, K*d_hid)
        out = self.predict(h)  # (N, d_out)
        return out
