"""Metapath Aggregated Graph Neural Network (MAGNN)

* 论文链接：https://arxiv.org/pdf/2002.01680
* 官方代码：https://github.com/cynricfu/MAGNN
"""
from collections import defaultdict

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import edge_softmax

from pytorch_tutorial.gnn.magnn.metapath_instance_encoder import get_metapath_instance_encoder
from pytorch_tutorial.gnn.magnn.utils import metapath_based_graph, metapath_instance_feat


class IntraMetapathAggregation(nn.Module):

    def __init__(
            self, in_dim, hidden_dim, num_heads, metapath_instance_encoder,
            attn_drop, negative_slope, activation=None
    ):
        """元路径内的聚集

        针对一种顶点类型和一条首尾为该类型的元路径，将每个目标顶点所有给定元路径的实例编码为一个向量

        :param in_dim: 输入特征维数
        :param hidden_dim: 隐含特征维数
        :param num_heads: 注意力头数K
        :param metapath_instance_encoder: 元路径实例编码器名称
        :param attn_drop: 注意力权重dropout比例
        :param negative_slope: LeakyReLU负斜率
        :param activation: 用于输出特征的激活函数
        """
        super().__init__()
        self.metapath_instance_encoder = get_metapath_instance_encoder(
            metapath_instance_encoder, in_dim, num_heads * hidden_dim
        )

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden_dim)))
        self.attn_r = nn.Linear(in_dim, num_heads, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain)

    def forward(self, g, node_feat, edge_feat):
        """
        :param g: DGLGraph 基于给定元路径的邻居构成的图，每条边表示一个元路径实例
        :param node_feat: tensor(N, d_in) 输入顶点特征，N为g的顶点个数
        :param edge_feat: tensor(E, L, d_in) 元路径实例特征（由中间顶点的特征组成），E为g的边数，L为元路径长度
        :return: tensor(N, K, d_hid) 输出顶点特征，K为注意力头数
        """
        # 与GAT/HAN顶点层次的注意力的区别：注意力对象由基于元路径的邻居改为元路径实例，考虑了元路径实例的中间顶点
        with g.local_scope():
            edge_feat = self.metapath_instance_encoder(edge_feat)  # (E, L, d_in) -> (E, K*d_hid)
            edge_feat = edge_feat.view(-1, self.num_heads, self.feat_dim)  # (E, K, d_hid)
            # a^T (h_p || h_v) = (a_l^T || a_r^T) (h_p || h_v) = a_l^T h_p + a_r^T h_v = el + er
            el = (edge_feat * self.attn_l).sum(dim=-1).unsqueeze(dim=-1)  # (E, K, 1)
            er = self.attn_r(node_feat).unsqueeze(dim=-1)  # (N, K, 1)
            g.edata.update({'ft': edge_feat, 'el': el})
            g.dstdata['er'] = er
            g.apply_edges(fn.e_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = self.attn_drop(edge_softmax(g, e))  # (E, K, 1)

            # 消息传递，相当于聚集
            g.update_all(lambda edges: {'m': edges.data['ft'] * edges.data['a']}, fn.sum('m', 'ft'))
            ret = g.ndata['ft']
            if self.activation:
                ret = self.activation(ret)
            return ret


class InterMetapathAggregation(nn.Module):

    def __init__(self, in_dim, attn_hidden_dim):
        """元路径间的聚集

        针对一种顶点类型和所有首尾为该类型的元路径，将每个顶点关于所有元路径的向量组合起来

        :param in_dim: 顶点关于元路径的向量维数（对应元路径内的聚集模块的K*d_hid）
        :param attn_hidden_dim: 中间隐含向量维数
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, attn_hidden_dim)  # 论文中(5)式的M_A和b_A
        self.fc2 = nn.Linear(attn_hidden_dim, 1, bias=False)  # 论文中(6)式的q_A

    def forward(self, metapath_feat):
        """
        :param metapath_feat: tensor(N, M, d_in) 每个顶点关于所有元路径的向量，N为顶点数，M为元路径个数
        :return: tensor(N, d_in) 聚集后的顶点嵌入
        """
        # 与HAN语义层次的注意力完全相同
        s = torch.tanh(self.fc1(metapath_feat)).mean(dim=0)  # (M, d_hid)
        e = self.fc2(s)  # (M, 1)
        beta = e.softmax(dim=0)  # (M)
        beta = beta.reshape((1, beta.shape[0], 1))  # (1, M, 1)
        return (beta * metapath_feat).sum(dim=1)


class MAGNNLayerNtypeSpecific(nn.Module):

    def __init__(
            self, num_metapaths, in_dim, hidden_dim, num_heads, metapath_instance_encoder,
            attn_hidden_dim, attn_drop, negative_slope
    ):
        """特定顶点类型的MAGNN层

        针对一种顶点类型和所有首尾为该类型的元路径，分别对每条元路径做元路径内的聚集，之后做元路径间的聚集

        :param num_metapaths: 元路径个数
        :param in_dim: 输入特征维数
        :param hidden_dim: 隐含特征维数
        :param num_heads: 注意力头数
        :param metapath_instance_encoder: 元路径实例编码器名称
        :param attn_hidden_dim: 元路径间的聚集中间隐含向量维数
        :param attn_drop: 注意力权重dropout比例
        :param negative_slope: LeakyReLU负斜率
        """
        super().__init__()
        self.intra_metapath_aggregations = nn.ModuleList([
            IntraMetapathAggregation(
                in_dim, hidden_dim, num_heads, metapath_instance_encoder,
                attn_drop, negative_slope, F.elu
            )
            for _ in range(num_metapaths)
        ])
        self.inter_metapath_aggregation = InterMetapathAggregation(num_heads * in_dim, attn_hidden_dim)

    def forward(self, gs, node_feat, edge_feat_name):
        """
        :param gs: List[DGLGraph] 基于每条元路径的邻居构成的图
        :param node_feat: tensor(N, d_in) 输入顶点特征，N为给定类型的顶点个数
        :param edge_feat_name: str 元路径特征所在的边属性名称
        :return: tensor(N, K*d_hid) 最终顶点嵌入，K为注意力头数
        """
        metapath_embeddings = [
            self.intra_metapath_aggregations[i](g, node_feat, g.edata[edge_feat_name]).flatten(start_dim=1)
            for i, g in enumerate(gs)
        ]  # List[tensor(N, K*d_hid)]
        metapath_embeddings = torch.stack(metapath_embeddings, dim=1)  # (N, M, K*d_hid)
        return self.inter_metapath_aggregation(metapath_embeddings)  # (N, K*d_hid)


class MAGNNLayer(nn.Module):

    def __init__(
            self, num_metapaths, in_dim, hidden_dim, out_dim, num_heads, metapath_instance_encoder,
            attn_hidden_dim, attn_drop, negative_slope
    ):
        """MAGNN层

        针对多种顶点类型、每种顶点类型对应的所有首尾为该类型的元路径

        :param num_metapaths: Dict[str, int] 顶点类型到其对应的元路径个数的映射
        :param in_dim: 输入特征维数
        :param hidden_dim: 隐含特征维数
        :param out_dim: 输出特征维数
        :param num_heads: 注意力头数
        :param metapath_instance_encoder: 元路径实例编码器名称
        :param attn_hidden_dim: 元路径间的聚集中间隐含向量维数
        :param attn_drop: 注意力权重dropout比例
        :param negative_slope: LeakyReLU负斜率
        """
        super().__init__()
        self.ntype_specific_layers = nn.ModuleDict({
            ntype: MAGNNLayerNtypeSpecific(
                num_metapaths[ntype], in_dim, hidden_dim, num_heads, metapath_instance_encoder,
                attn_hidden_dim, attn_drop, negative_slope
            )
            for ntype in num_metapaths
        })
        self.fc = nn.Linear(num_heads * hidden_dim, out_dim)

    def forward(self, gs, node_feats, edge_feat_name):
        """
        :param gs: Dict[str, List[DGLGraph]] 顶点类型到其对应的基于每条元路径的邻居构成的图的映射
        :param node_feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入顶点特征的映射，N_i为对应类型的顶点个数
        :param edge_feat_name: str 元路径特征所在的边属性名称
        :return: Dict[str, (tensor(N_i, K*d_hid), tensor(N_i, d_out))] 顶点类型到(最终顶点嵌入,预测值)的映射
        """
        outputs = {}
        for ntype in gs:
            embeddings = self.ntype_specific_layers[ntype](gs[ntype], node_feats[ntype], edge_feat_name)
            logits = self.fc(embeddings)
            outputs[ntype] = (embeddings, logits)
        return outputs


class MAGNN(nn.Module):

    def __init__(
            self, num_layers, metapaths, in_dims, trans_dim, out_dim, num_heads,
            metapath_instance_encoder, attn_hidden_dim, feat_drop, attn_drop, negative_slope
    ):
        """MAGNN模型，由特征转换、多个MAGNN层和输出层组成

        :param num_layers: int MAGNN层数
        :param metapaths: Dict[str, List[(str, str, str)]] 顶点类型到其对应的所有元路径的映射
        :param in_dims: Dict[str, int] 顶点类型到原始特征维数的映射
        :param trans_dim: 转换后的特征维数
        :param out_dim: 输出特征维数
        :param num_heads: 注意力头数
        :param metapath_instance_encoder: 元路径实例编码器名称
        :param attn_hidden_dim: 元路径间的聚集中间隐含向量维数
        :param feat_drop: 特征dropout比例
        :param attn_drop: 注意力权重dropout比例
        :param negative_slope: LeakyReLU负斜率
        """
        super().__init__()
        self.feat_trans = nn.ModuleDict({
            ntype: nn.Linear(in_dims[ntype], trans_dim) for ntype in in_dims
        })
        self.feat_drop = nn.Dropout(feat_drop)

        self.metapaths = metapaths
        num_metapaths = {ntype: len(metapaths[ntype]) for ntype in metapaths}
        self.layers = nn.ModuleList([
            MAGNNLayer(
                num_metapaths, trans_dim, trans_dim, trans_dim, num_heads,
                metapath_instance_encoder, attn_hidden_dim, attn_drop, negative_slope
            )
            for _ in range(num_layers - 1)
        ])
        self.layers.append(MAGNNLayer(
            num_metapaths, trans_dim, trans_dim, out_dim, num_heads,
            metapath_instance_encoder, attn_hidden_dim, attn_drop, negative_slope
        ))

        self._metapath_based_graphs = defaultdict(list)  # Dict[str, List[DGLGraph]]
        self._metapath_instances = defaultdict(list)  # Dict[str, List[List[List[int]]]]

    def forward(self, g, node_feat_name):
        """
        :param g: DGLHeteroGraph 异构图
        :param node_feat_name: 顶点特征名称
        :return: Dict[str, (tensor(N_i, K*d_trans), tensor(N_i, d_out))] 顶点类型到(最终顶点嵌入,预测值)的映射
        """
        with g.local_scope():
            if not self._metapath_based_graphs:
                self._build_metapath_based_graphs(g)
            # 顶点特征转换
            embeddings = {}  # Dict[str, tensor(N_i, d_trans)]
            trans_feat_name = 'trans' + node_feat_name
            for ntype in self.feat_trans:
                embeddings[ntype] = g.nodes[ntype].data[trans_feat_name] = \
                    self.feat_drop(self.feat_trans[ntype](g.nodes[ntype].data[node_feat_name]))

            # 计算元路径实例特征
            for ntype in self.metapaths:
                for i in range(len(self.metapaths[ntype])):
                    self._metapath_based_graphs[ntype][i].edata[trans_feat_name] = \
                        metapath_instance_feat(g, self.metapaths[ntype][i], self._metapath_instances[ntype][i], trans_feat_name)

            for i in range(len(self.layers) - 1):
                outputs = self.layers[i](self._metapath_based_graphs, embeddings, trans_feat_name)
                # Dict[str, tensor(N_i, d_trans)]
                embeddings = {ntype: logits for ntype, (_, logits) in outputs.items()}
            return self.layers[-1](self._metapath_based_graphs, embeddings, trans_feat_name)

    def _build_metapath_based_graphs(self, g):
        """构造异构图基于每条元路径的邻居构成的图及元路径实例。"""
        for ntype in self.metapaths:
            for metapath in self.metapaths[ntype]:
                mg, instances = metapath_based_graph(g, metapath)
                self._metapath_based_graphs[ntype].append(mg)
                self._metapath_instances[ntype].append(instances)
