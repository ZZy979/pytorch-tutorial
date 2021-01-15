"""Graph Attention Networks (GAT)

* 论文链接：https://arxiv.org/abs/1710.10903
* DGL教程：https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
* DGL实现：https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat
"""
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv


class GAT(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout, activation=None):
        """GAT模型

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: List[int] 每一层的注意力头数，长度等于层数
        :param dropout: float Dropout概率
        :param activation: callable, optional 输出层激活函数
        :raise ValueError: 如果层数（即num_heads的长度）小于2
        """
        super().__init__()
        num_layers = len(num_heads)
        if num_layers < 2:
            raise ValueError('层数至少为2，实际为{}'.format(num_layers))
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(
            in_dim, hidden_dim, num_heads[0], dropout, dropout, activation=F.elu
        ))
        for i in range(1, num_layers - 1):
            self.layers.append(GATConv(
                num_heads[i - 1] * hidden_dim, hidden_dim, num_heads[i], dropout, dropout,
                activation=F.elu
            ))
        self.layers.append(GATConv(
            num_heads[-2] * hidden_dim, out_dim, num_heads[-1], dropout, dropout,
            activation=activation
        ))

    def forward(self, g, h):
        """
        :param g: DGLGraph 同构图
        :param h: tensor(N, d_in) 输入特征，N为g的顶点数
        :return: tensor(N, d_out) 输出顶点特征，K为注意力头数
        """
        for i in range(len(self.layers) - 1):
            h = self.layers[i](g, h).flatten(start_dim=1)  # (N, K, d_hid) -> (N, K*d_hid)
        h = self.layers[-1](g, h).mean(dim=1)  # (N, K, d_out) -> (N, d_out)
        return h
