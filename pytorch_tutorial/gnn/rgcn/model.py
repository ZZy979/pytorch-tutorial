"""Relational Graph Convolutional Network (R-GCN)

* 论文链接：https://arxiv.org/abs/1703.06103
* DGL教程：https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html
* DGL实现：https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GraphConv, WeightBasis


class RelGraphConv(nn.Module):

    def __init__(
            self, in_dim, out_dim, rel_names, num_bases=None,
            weight=True, self_loop=True, activation=None):
        """R-GCN层

        :param in_dim: 输入特征维数
        :param out_dim: 输出特征维数
        :param rel_names: List[str] 关系名称
        :param num_bases: int, optional 基的个数，默认使用关系个数
        :param weight: bool, optional 是否进行线性变换，默认为True
        :param self_loop: 是否包括自环消息，默认为True
        :param activation: callable, optional 激活函数，默认为None
        """
        super().__init__()
        self.rel_names = rel_names
        self.self_loop = self_loop
        self.activation = activation

        self.conv = HeteroGraphConv({
            rel: GraphConv(in_dim, out_dim, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = weight and 0 < num_bases < len(rel_names)
        if self.use_weight:
            if self.use_basis:
                self.basis = WeightBasis((in_dim, out_dim), num_bases, len(rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(rel_names), in_dim, out_dim))
                nn.init.xavier_uniform_(self.weight, nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.self_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(self.self_weight, nn.init.calculate_gain('relu'))

    def forward(self, g, inputs):
        """
        :param g: DGLHeteroGraph 异构图
        :param inputs: Dict[str, tensor(N_i, d_in)] 顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到输出特征的映射
        """
        with g.local_scope():
            if self.use_weight:
                weight = self.basis() if self.use_basis else self.weight  # (R, d_in, d_out)
                kwargs = {rel: {'weight': weight[i]} for i, rel in enumerate(self.rel_names)}
            else:
                kwargs = {}
            hs = self.conv(g, inputs, mod_kwargs=kwargs)  # Dict[ntype, (N_i, d_out)]
            for ntype in hs:
                if self.self_loop:
                    hs[ntype] += torch.matmul(inputs[ntype], self.self_weight)
                if self.activation:
                    hs[ntype] = self.activation(hs[ntype])
            return hs


class RelGraphEmbed(nn.Module):

    def __init__(self, num_nodes, embed_dim):
        """用于获取无特征异构图的输入顶点特征的嵌入层

        :param num_nodes: Dict[str, int] 顶点类型到顶点数的映射
        :param embed_dim: 特征维数d
        """
        super().__init__()
        self.embeds = nn.ModuleDict({
            ntype: nn.Embedding(num_nodes[ntype], embed_dim) for ntype in num_nodes
        })
        self.reset_parameters()

    def reset_parameters(self):
        # normal_ -> xavier_uniform_ => acc 0.8 -> 0.97?
        for k in self.embeds:
            nn.init.xavier_uniform_(self.embeds[k].weight, gain=nn.init.calculate_gain('relu'))

    def forward(self):
        """
        :return: Dict[str, tensor(N_i, d)] 顶点类型到顶点特征的映射
        """
        return {k: self.embeds[k].weight for k in self.embeds}


class EntityClassification(nn.Module):

    def __init__(self, num_nodes, hidden_dim, out_dim, rel_names, num_bases=None, self_loop=True):
        """两层R-GCN实体分类模型

        :param num_nodes: Dict[str, int] 顶点类型到顶点数的映射
        :param hidden_dim: 隐含特征维数
        :param out_dim: 输出特征维数
        :param rel_names: List[str] 关系名称
        :param num_bases: int, optional 基的个数，默认使用关系个数
        :param self_loop: 是否包括自环消息，默认为True
        """
        super().__init__()
        self.embed = RelGraphEmbed(num_nodes, hidden_dim)
        self.conv1 = RelGraphConv(hidden_dim, hidden_dim, rel_names, num_bases, False, self_loop, F.relu)
        self.conv2 = RelGraphConv(hidden_dim, out_dim, rel_names, num_bases, True, self_loop)

    def forward(self, g):
        """
        :param g: DGLHeteroGraph 异构图
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到顶点嵌入的映射
        """
        h = self.embed()
        h = self.conv1(g, h)  # Dict[ntype, (N_i, d_hid)]
        return self.conv2(g, h)
