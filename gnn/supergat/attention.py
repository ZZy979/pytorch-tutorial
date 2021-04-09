import math

import dgl.function as fn
import torch
import torch.nn as nn


class GraphAttention(nn.Module):
    """图注意力模块，用于计算顶点邻居的重要性"""

    def __init__(self, out_dim, num_heads):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

    def forward(self, g, feat_src, feat_dst):
        """
        :param g: DGLGraph 同构图
        :param feat_src: tensor(N_src, K, d_out) 起点特征
        :param feat_dst: tensor(N_dst, K, d_out) 终点特征
        :return: tensor(E, K, 1) 所有顶点对的邻居重要性
        """
        raise NotImplementedError


class GATOriginalAttention(GraphAttention):
    """原始GAT注意力"""

    def __init__(self, out_dim, num_heads):
        super().__init__(out_dim, num_heads)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, g, feat_src, feat_dst):
        el = (feat_src * self.attn_l).sum(dim=-1, keepdim=True)  # (N_src, K, 1)
        er = (feat_dst * self.attn_r).sum(dim=-1, keepdim=True)  # (N_dst, K, 1)
        g.srcdata['el'] = el
        g.dstdata['er'] = er
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        return g.edata.pop('e')


class DotProductAttention(GraphAttention):
    """点积注意力"""

    def forward(self, g, feat_src, feat_dst):
        g.srcdata['ft'] = feat_src
        g.dstdata['ft'] = feat_dst
        g.apply_edges(lambda edges: {
            'e': torch.sum(edges.src['ft'] * edges.dst['ft'], dim=-1, keepdim=True)
        })
        return g.edata.pop('e')


class ScaledDotProductAttention(DotProductAttention):

    def forward(self, g, feat_src, feat_dst):
        return super().forward(g, feat_src, feat_dst) / math.sqrt(self.out_dim)


class MixedGraphAttention(GATOriginalAttention, DotProductAttention):

    def forward(self, g, feat_src, feat_dst):
        return GATOriginalAttention.forward(self, g, feat_src, feat_dst) \
               * torch.sigmoid(DotProductAttention.forward(self, g, feat_src, feat_dst))


GRAPH_ATTENTIONS = {
    'GO': GATOriginalAttention,
    'DP': DotProductAttention,
    'SD': ScaledDotProductAttention,
    'MX': MixedGraphAttention
}


def get_graph_attention(attn_type, out_dim, num_heads):
    if attn_type in GRAPH_ATTENTIONS:
        return GRAPH_ATTENTIONS[attn_type](out_dim, num_heads)
    else:
        raise ValueError('非法图注意力类型{}，可选项为{}'.format(attn_type, list(GRAPH_ATTENTIONS.keys())))
