"""How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision (SuperGAT)

* 论文链接：https://openreview.net/pdf?id=Wi5KUNlqWty
* 官方代码：https://github.com/dongkwan-kim/SuperGAT
"""
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import edge_softmax

from gnn.supergat.attention import get_graph_attention
from gnn.supergat.neg_sampler import RatioNegativeSampler


class SuperGATConv(nn.Module):

    def __init__(
            self, in_dim, out_dim, num_heads, attn_type, neg_sample_ratio=0.5,
            feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, activation=None):
        """SuperGAT层，自监督任务是连接预测

        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param attn_type: str 注意力类型，可选择GO, DP, SD和MX
        :param neg_sample_ratio: float, optional 负样本边数量占正样本边数量的比例，默认0.5
        :param feat_drop: float, optional 输入特征Dropout概率，默认为0
        :param attn_drop: float, optional 注意力权重Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        :param activation: callable, optional 用于输出特征的激活函数，默认为None
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.attn = get_graph_attention(attn_type, out_dim, num_heads)
        self.neg_sampler = RatioNegativeSampler(neg_sample_ratio)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

    def forward(self, g, feat):
        """
        :param g: DGLGraph 同构图
        :param feat: tensor(N_src, d_in) 输入顶点特征
        :return: tensor(N_dst, K, d_out) 输出顶点特征
        """
        with g.local_scope():
            feat_src = self.fc(self.feat_drop(feat)).view(-1, self.num_heads, self.out_dim)
            feat_dst = feat_src[:g.num_dst_nodes()] if g.is_block else feat_src
            e = self.leaky_relu(self.attn(g, feat_src, feat_dst))  # (E, K, 1)
            g.edata['a'] = self.attn_drop(edge_softmax(g, e))  # (E, K, 1)
            g.srcdata['ft'] = feat_src
            # 消息传递
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            out = g.dstdata['ft']  # (N_dst, K, d_out)

            if self.training:
                # 负采样
                neg_g = dgl.graph(
                    self.neg_sampler(g, list(range(g.num_edges()))), num_nodes=g.num_nodes(),
                    device=g.device
                )
                neg_e = self.attn(neg_g, feat_src, feat_src)  # (E', K, 1)
                self.attn_x = torch.cat([e, neg_e]).squeeze(dim=-1).mean(dim=1)  # (E+E',)
                self.attn_y = torch.cat([torch.ones(e.shape[0]), torch.zeros(neg_e.shape[0])]) \
                    .to(self.attn_x.device)

            if self.activation:
                out = self.activation(out)
            return out

    def get_attn_loss(self):
        """返回自监督注意力损失（即连接预测损失）"""
        if self.training:
            return F.binary_cross_entropy_with_logits(self.attn_x, self.attn_y)
        else:
            return torch.tensor(0.)


class SuperGAT(nn.Module):

    def __init__(
            self, in_dim, hidden_dim, out_dim, num_heads, attn_type, neg_sample_ratio=0.5,
            feat_drop=0.0, attn_drop=0.0, negative_slope=0.2):
        """两层SuperGAT模型

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param attn_type: str 注意力类型，可选择GO, DP, SD和MX
        :param neg_sample_ratio: float, optional 负样本边数量占正样本边数量的比例，默认0.5
        :param feat_drop: float, optional 输入特征Dropout概率，默认为0
        :param attn_drop: float, optional 注意力权重Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        """
        super().__init__()
        self.conv1 = SuperGATConv(
            in_dim, hidden_dim, num_heads, attn_type, neg_sample_ratio,
            feat_drop, attn_drop, negative_slope, F.elu
        )
        self.conv2 = SuperGATConv(
            num_heads * hidden_dim, out_dim, num_heads, attn_type, neg_sample_ratio,
            0, attn_drop, negative_slope
        )

    def forward(self, g, feat):
        """
        :param g: DGLGraph 同构图
        :param feat: tensor(N, d_in) 输入顶点特征
        :return: tensor(N, d_out), tensor(1) 输出顶点特征和自监督注意力损失
        """
        h = self.conv1(g, feat).flatten(start_dim=1)  # (N, K, d_hid) -> (N, K*d_hid)
        h = self.conv2(g, h).mean(dim=1)  # (N, K, d_out) -> (N, d_out)
        return h, self.conv1.get_attn_loss() + self.conv2.get_attn_loss()
