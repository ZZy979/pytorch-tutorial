"""Relational Graph Convolutional Network (R-GCN)

论文链接：https://arxiv.org/abs/1703.06103
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv


class DistMult(nn.Module):

    def __init__(self, num_rels, feat_dim):
        """知识图谱嵌入模型DistMult

        :param num_rels: int 关系个数
        :param feat_dim: int 嵌入维数
        """
        super().__init__()
        self.w_relations = nn.Parameter(torch.Tensor(num_rels, feat_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_relations, gain=nn.init.calculate_gain('relu'))

    def forward(self, embed, head, rel, tail):
        """
        :param embed: tensor(N, d) 实体嵌入
        :param head: tensor(*) 头实体
        :param rel: tensor(*) 关系
        :param tail: tensor(*) 尾实体
        :return: tensor(*) 三元组得分
        """
        # e_h^T R_r e_t = (e_h * diag(R_r)) e_t
        return torch.sum(embed[head] * self.w_relations[rel] * embed[tail], dim=1)


class LinkPrediction(nn.Module):

    def __init__(
            self, num_nodes, hidden_dim, num_rels, num_layers=2,
            regularizer='basis', num_bases=None, dropout=0.0):
        """R-GCN连接预测模型
        
        :param num_nodes: int 顶点（实体）数
        :param hidden_dim: int 隐含特征维数
        :param num_rels: int 关系个数
        :param num_layers: int, optional R-GCN层数，默认为2
        :param regularizer: str, 'basis'/'bdd' 权重正则化方法，默认为'basis'
        :param num_bases: int, optional 基的个数，默认使用关系个数
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.embed = nn.Embedding(num_nodes, hidden_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(RelGraphConv(
                hidden_dim, hidden_dim, num_rels, regularizer, num_bases,
                activation=F.relu if i < num_layers - 1 else None,
                self_loop=True, low_mem=True, dropout=dropout
            ))
        self.score = DistMult(num_rels, hidden_dim)

    def forward(self, g, etypes):
        """
        :param g: DGLGraph 同构图
        :param etypes: tensor(|E|) 边类型
        :return: tensor(N, d_hid) 顶点嵌入
        """
        h = self.embed.weight  # (N, d_hid)
        for layer in self.layers:
            h = layer(g, h, etypes)
        return h

    def calc_score(self, embed, triplets):
        """计算三元组得分

        :param embed: tensor(N, d_hid) 顶点（实体）嵌入
        :param triplets: (tensor(*), tensor(*), tensor(*)) 三元组(head, tail, relation)
        :return: tensor(*) 三元组得分
        """
        # 三元组的relation放在最后是为了与DGLGraph.find_edges的返回格式保持一致
        head, tail, rel = triplets
        return self.score(embed, head, rel, tail)
