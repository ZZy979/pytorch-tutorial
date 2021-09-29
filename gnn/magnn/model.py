"""Metapath Aggregated Graph Neural Network (MAGNN)

论文链接：https://arxiv.org/pdf/2002.01680
"""
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import edge_softmax

from .encoder import get_encoder
from ..utils import metapath_instance_feat


class IntraMetapathAggregation(nn.Module):

    def __init__(
            self, in_dim, out_dim, num_heads, encoder,
            attn_drop=0.0, negative_slope=0.01, activation=None):
        """元路径内的聚集

        针对一种顶点类型和 **一个** 首尾为该类型的元路径，将每个目标顶点所有给定元路径的实例编码为一个向量

        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param encoder: str 元路径实例编码器名称
        :param attn_drop: float, optional 注意力权重Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.01
        :param activation: callable, optional 用于输出特征的激活函数，默认为None
        """
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.encoder = get_encoder(encoder, in_dim, num_heads * out_dim)
        self.attn_l = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        self.attn_r = nn.Linear(in_dim, num_heads, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.attn_l, nn.init.calculate_gain('relu'))

    def forward(self, g, node_feat, edge_feat):
        """
        :param g: DGLGraph 基于给定元路径的邻居组成的图，每条边表示一个元路径实例
        :param node_feat: tensor(N, d_in) 输入顶点特征，N为g的终点个数
        :param edge_feat: tensor(E, L, d_in) 元路径实例特征（由中间顶点的特征组成），E为g的边数，L为元路径长度
        :return: tensor(N, K, d_out) 输出顶点特征，K为注意力头数
        """
        # 与GAT/HAN顶点层次的注意力的区别：注意力对象由基于元路径的邻居改为元路径实例，考虑了元路径实例的中间顶点
        with g.local_scope():
            edge_feat = self.encoder(edge_feat)  # (E, L, d_in) -> (E, K*d_out)
            edge_feat = edge_feat.view(-1, self.num_heads, self.out_dim)  # (E, K, d_out)
            # a^T (h_p || h_v) = (a_l^T || a_r^T) (h_p || h_v) = a_l^T h_p + a_r^T h_v = el + er
            el = (edge_feat * self.attn_l).sum(dim=-1).unsqueeze(dim=-1)  # (E, K, 1)
            er = self.attn_r(node_feat).unsqueeze(dim=-1)  # (N, K, 1)
            g.edata.update({'ft': edge_feat, 'el': el})
            g.dstdata['er'] = er
            g.apply_edges(fn.e_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = self.attn_drop(edge_softmax(g, e))  # (E, K, 1)

            # 消息传递
            g.update_all(lambda edges: {'m': edges.data['ft'] * edges.data['a']}, fn.sum('m', 'ft'))
            ret = g.dstdata['ft']
            if self.activation:
                ret = self.activation(ret)
            return ret


class InterMetapathAggregation(nn.Module):

    def __init__(self, in_dim, attn_hidden_dim):
        """元路径间的聚集

        针对一种顶点类型和所有首尾为该类型的元路径，将每个顶点关于所有元路径的嵌入组合起来

        :param in_dim: int 顶点关于元路径的嵌入维数（对应元路径内的聚集模块的输出维数）
        :param attn_hidden_dim: int 中间隐含向量维数
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, attn_hidden_dim)  # 论文中(5)式的M_A和b_A
        self.fc2 = nn.Linear(attn_hidden_dim, 1, bias=False)  # 论文中(6)式的q_A

    def forward(self, z):
        """
        :param z: tensor(N, M, d_in) 每个顶点关于所有元路径的嵌入，N为顶点数，M为元路径个数
        :return: tensor(N, d_in) 聚集后的顶点嵌入
        """
        # 与HAN语义层次的注意力完全相同
        s = torch.tanh(self.fc1(z)).mean(dim=0)  # (N, M, d_in) -> (M, d_in)
        e = self.fc2(s)  # (M, 1)
        beta = e.softmax(dim=0)  # (M)
        beta = beta.reshape((1, -1, 1))  # (1, M, 1)
        z = (beta * z).sum(dim=1)  # (N, d_in)
        return z


class MAGNNLayerNtypeSpecific(nn.Module):

    def __init__(
            self, num_metapaths, in_dim, out_dim, num_heads, encoder,
            attn_hidden_dim=128, attn_drop=0.0):
        """特定顶点类型的MAGNN层

        针对一种顶点类型和所有首尾为该类型的元路径，分别对每条元路径做元路径内的聚集，之后做元路径间的聚集

        :param num_metapaths: int 元路径个数
        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param encoder: str 元路径实例编码器名称
        :param attn_hidden_dim: int, optional 元路径间的聚集中间隐含向量维数，默认为128
        :param attn_drop: float, optional 注意力权重Dropout概率，默认为0
        """
        super().__init__()
        self.intra_metapath_aggs = nn.ModuleList([
            IntraMetapathAggregation(
                in_dim, out_dim, num_heads, encoder, attn_drop, activation=F.elu
            ) for _ in range(num_metapaths)
        ])
        self.inter_metapath_agg = InterMetapathAggregation(num_heads * out_dim, attn_hidden_dim)

    def forward(self, gs, node_feat, edge_feat_name):
        """
        :param gs: List[DGLGraph] 基于每条元路径的邻居组成的图
        :param node_feat: tensor(N, d_in) 输入顶点特征，N为给定类型的顶点个数
        :param edge_feat_name: str 元路径实例特征所在的边属性名称
        :return: tensor(N, K*d_out) 最终顶点嵌入，K为注意力头数
        """
        if gs[0].is_block:
            node_feat = node_feat[gs[0].dstdata[dgl.NID]]
        metapath_embeds = [
            agg(g, node_feat, g.edata[edge_feat_name]).flatten(start_dim=1)  # tensor(N, K*d_out)
            for agg, g in zip(self.intra_metapath_aggs, gs)
        ]
        metapath_embeds = torch.stack(metapath_embeds, dim=1)  # (N, M, K*d_out)
        return self.inter_metapath_agg(metapath_embeds)  # (N, K*d_out)


class MAGNNLayer(nn.Module):

    def __init__(self, metapaths, in_dim, out_dim, num_heads, encoder, attn_drop=0.0):
        """MAGNN层

        :param metapaths: Dict[str, List[List[str]]] 顶点类型到其对应的元路径的映射，元路径表示为顶点类型列表
        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param encoder: str 元路径实例编码器名称
        :param attn_drop: float, optional 注意力权重Dropout概率，默认为0
        """
        super().__init__()
        self.metapaths = metapaths
        self.layers = nn.ModuleDict({
            ntype: MAGNNLayerNtypeSpecific(
                len(metapaths[ntype]), in_dim, out_dim, num_heads, encoder, attn_drop=attn_drop
            ) for ntype in metapaths
        })
        self.fc = nn.Linear(num_heads * out_dim, out_dim)

    def forward(self, gs, node_feats):
        """
        :param gs: Dict[str, List[DGLGraph]] 顶点类型到其对应的基于每条元路径的邻居组成的图的映射
        :param node_feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入顶点特征的映射，N_i为对应类型的顶点个数
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到最终顶点嵌入的映射
        """
        self._calc_metapath_instance_feat(gs, node_feats)
        return {
            ntype: self.fc(self.layers[ntype](gs[ntype], node_feats[ntype], 'feat'))
            for ntype in gs
        }

    def _calc_metapath_instance_feat(self, gs, node_feats):
        for ntype in self.metapaths:
            for g, metapath in zip(gs[ntype], self.metapaths[ntype]):
                g.edata['feat'] = metapath_instance_feat(metapath, node_feats, g.edata['inst'])


# 如果使用minibatch训练则模型只能是一层，因为难以保证每次采样的顶点满足计算元路径实例特征的需要，此时只有计算待分类类型顶点的嵌入才有意义
# 反之，如果使用多层模型则只能使用全图训练，此时可以计算多种类型顶点的嵌入（作为下一层的输入特征）
class MAGNNMinibatch(nn.Module):

    def __init__(self, ntype, metapaths, in_dims, hidden_dim, out_dim, num_heads, encoder, dropout=0.0):
        """使用minibatch训练的MAGNN模型，由特征转换、一个MAGNN层和输出层组成。

        仅针对目标顶点类型及其对应的元路径计算嵌入，其他顶点类型仅使用输入特征

        :param ntype: str 目标顶点类型
        :param metapaths: List[List[str]] 目标顶点类型对应的元路径列表，元路径表示为顶点类型列表
        :param in_dims: Dict[str, int] 所有顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: 注意力头数K
        :param encoder: str 元路径实例编码器名称
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.ntype = ntype
        self.feat_trans = nn.ModuleDict({
            ntype: nn.Linear(in_dims[ntype], hidden_dim) for ntype in in_dims
        })
        self.feat_drop = nn.Dropout(dropout)
        self.magnn = MAGNNLayer(
            {self.ntype: metapaths}, hidden_dim, out_dim, num_heads, encoder, dropout
        )

    def forward(self, blocks, node_feats):
        """
        :param blocks: List[DGLBlock] 目标顶点类型对应的每条元路径的邻居组成的图(block)
        :param node_feats: Dict[str, tensor(N_i, d_in)] 所有顶点类型到输入顶点特征的映射，N_i为对应类型的顶点个数
        :return: tensor(N, d_out) 目标顶点类型的最终嵌入
        """
        hs = {
            ntype: self.feat_drop(trans(node_feats[ntype]))
            for ntype, trans in self.feat_trans.items()
        }  # Dict[str, tensor(N_i, d_hid)]
        out = self.magnn({self.ntype: blocks}, hs)[self.ntype]  # tensor(N, d_out)
        return out


class MAGNNMultiLayer(nn.Module):

    def __init__(self, num_layers, metapaths, in_dims, hidden_dim, out_dim, num_heads, encoder, dropout=0.0):
        """多层MAGNN模型，由特征转换、多个MAGNN层和输出层组成。

        :param num_layers: int MAGNN层数
        :param metapaths: Dict[str, List[List[str]]] 顶点类型到元路径列表的映射，元路径表示为顶点类型列表
        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: 注意力头数K
        :param encoder: str 元路径实例编码器名称
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.feat_trans = nn.ModuleDict({
            ntype: nn.Linear(in_dims[ntype], hidden_dim) for ntype in in_dims
        })
        self.feat_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            MAGNNLayer(metapaths, hidden_dim, hidden_dim, num_heads, encoder, dropout)
            for _ in range(num_layers - 1)
        ])
        self.layers.append(MAGNNLayer(
            metapaths, hidden_dim, out_dim, num_heads, encoder, dropout
        ))

    def forward(self, gs, node_feats):
        """
        :param gs: Dict[str, List[DGLGraph]] 顶点类型到其对应的基于每条元路径的邻居组成的图的映射
        :param node_feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入顶点特征的映射，N_i为对应类型的顶点个数
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到最终顶点嵌入的映射
        """
        hs = {
            ntype: self.feat_drop(trans(node_feats[ntype]))
            for ntype, trans in self.feat_trans.items()
        }  # Dict[str, tensor(N_i, d_hid)]
        for i in range(len(self.layers) - 1):
            hs = self.layers[i](gs, hs)  # Dict[str, tensor(N_i, d_hid)]
            hs = {ntype: F.elu(h) for ntype, h in hs.items()}
        return self.layers[-1](gs, hs)
