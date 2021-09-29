"""Heterogeneous Graph Representation Learning with Relation Awareness (R-HGNN)

论文链接：https://arxiv.org/pdf/2105.11122
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.utils.data import DataLoader
from tqdm import tqdm


class RelationGraphConv(nn.Module):

    def __init__(
            self, out_dim, num_heads, fc_src, fc_dst, fc_rel,
            feat_drop=0.0, negative_slope=0.2, activation=None):
        """特定关系的卷积

        针对一种关系（边类型）R=<stype, etype, dtype>，聚集关系R下的邻居信息，得到dtype类型顶点在关系R下的表示，
        注意力向量使用关系R的表示

        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param fc_src: nn.Linear(d_in, K*d_out) 源顶点特征转换模块
        :param fc_dst: nn.Linear(d_in, K*d_out) 目标顶点特征转换模块
        :param fc_rel: nn.Linear(d_rel, 2*K*d_out) 关系表示转换模块
        :param feat_drop: float, optional 输入特征Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        :param activation: callable, optional 用于输出特征的激活函数，默认为None
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.fc_src = fc_src
        self.fc_dst = fc_dst
        self.fc_rel = fc_rel
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

    def forward(self, g, feat, feat_rel):
        """
        :param g: DGLGraph 二分图（只包含一种关系）
        :param feat: tensor(N_src, d_in) or (tensor(N_src, d_in), tensor(N_dst, d_in)) 输入特征
        :param feat_rel: tensor(d_rel) 关系R的表示
        :return: tensor(N_dst, K*d_out) 目标顶点在关系R下的表示
        """
        with g.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, g)
            feat_src = self.fc_src(self.feat_drop(feat_src)).view(-1, self.num_heads, self.out_dim)
            feat_dst = self.fc_dst(self.feat_drop(feat_dst)).view(-1, self.num_heads, self.out_dim)
            attn = self.fc_rel(feat_rel).view(self.num_heads, 2 * self.out_dim)

            # a^T (z_u || z_v) = (a_l^T || a_r^T) (z_u || z_v) = a_l^T z_u + a_r^T z_v = el + er
            el = (feat_src * attn[:, :self.out_dim]).sum(dim=-1, keepdim=True)  # (N_src, K, 1)
            er = (feat_dst * attn[:, self.out_dim:]).sum(dim=-1, keepdim=True)  # (N_dst, K, 1)
            g.srcdata.update({'ft': feat_src, 'el': el})
            g.dstdata['er'] = er
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = edge_softmax(g, e)  # (E, K, 1)

            # 消息传递
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            ret = g.dstdata['ft'].view(-1, self.num_heads * self.out_dim)
            if self.activation:
                ret = self.activation(ret)
            return ret


class RelationCrossing(nn.Module):

    def __init__(self, out_dim, num_heads, rel_attn, dropout=0.0, negative_slope=0.2):
        """跨关系消息传递

        针对一种关系R=<stype, etype, dtype>，将dtype类型顶点在不同关系下的表示进行组合

        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param rel_attn: nn.Parameter(K, d) 关系R的注意力向量
        :param dropout: float, optional Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rel_attn = rel_attn
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, feats):
        """
        :param feats: tensor(N_R, N, K*d) dtype类型顶点在不同关系下的表示
        :return: tensor(N, K*d) 跨关系消息传递后dtype类型顶点在关系R下的表示
        """
        num_rel = feats.shape[0]
        if num_rel == 1:
            return feats.squeeze(dim=0)
        feats = feats.view(num_rel, -1, self.num_heads, self.out_dim)  # (N_R, N, K, d)
        attn_scores = (self.rel_attn * feats).sum(dim=-1, keepdim=True)
        attn_scores = F.softmax(self.leaky_relu(attn_scores), dim=0)  # (N_R, N, K, 1)
        out = (attn_scores * feats).sum(dim=0)  # (N, K, d)
        out = self.dropout(out.view(-1, self.num_heads * self.out_dim))  # (N, K*d)
        return out


class RelationFusing(nn.Module):

    def __init__(
            self, node_hidden_dim, rel_hidden_dim, num_heads,
            w_node, w_rel, dropout=0.0, negative_slope=0.2):
        """关系混合

        针对一种顶点类型，将该类型顶点在不同关系下的表示进行组合

        :param node_hidden_dim: int 顶点隐含特征维数
        :param rel_hidden_dim: int 关系隐含特征维数
        :param num_heads: int 注意力头数K
        :param w_node: Dict[str, tensor(K, d_node, d_node)] 边类型到顶点关于该关系的特征转换矩阵的映射
        :param w_rel: Dict[str, tensor(K, d_rel, d_node)] 边类型到关系的特征转换矩阵的映射
        :param dropout: float, optional Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        """
        super().__init__()
        self.node_hidden_dim = node_hidden_dim
        self.rel_hidden_dim = rel_hidden_dim
        self.num_heads = num_heads
        self.w_node = nn.ParameterDict(w_node)
        self.w_rel = nn.ParameterDict(w_rel)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, node_feats, rel_feats):
        """
        :param node_feats: Dict[str, tensor(N, K*d_node)] 边类型到顶点在该关系下的表示的映射
        :param rel_feats: Dict[str, tensor(K*d_rel)] 边类型到关系的表示的映射
        :return: tensor(N, K*d_node) 该类型顶点的最终嵌入
        """
        etypes = list(node_feats.keys())
        num_rel = len(node_feats)
        if num_rel == 1:
            return node_feats[etypes[0]]
        node_feats = torch.stack([node_feats[e] for e in etypes], dim=0) \
            .reshape(num_rel, -1, self.num_heads, self.node_hidden_dim)  # (N_R, N, K, d_node)
        rel_feats = torch.stack([rel_feats[e] for e in etypes], dim=0) \
            .reshape(num_rel, self.num_heads, self.rel_hidden_dim)  # (N_R, K, d_rel)
        w_node = torch.stack([self.w_node[e] for e in etypes], dim=0)  # (N_R, K, d_node, d_node)
        w_rel = torch.stack([self.w_rel[e] for e in etypes], dim=0)  # (N_R, K, d_rel, d_node)

        # hn[r, n, h] @= wn[r, h] => hn[r, n, h, i] = ∑(k) hn[r, n, h, k] * wn[r, h, k, i]
        node_feats = torch.einsum('rnhk,rhki->rnhi', node_feats, w_node)  # (N_R, N, K, d_node)
        # hr[r, h] @= wr[r, h] => hr[r, h, i] = ∑(k) hr[r, h, k] * wr[r, h, k, i]
        rel_feats = torch.einsum('rhk,rhki->rhi', rel_feats, w_rel)  # (N_R, K, d_node)

        attn_scores = (node_feats * rel_feats.unsqueeze(dim=1)).sum(dim=-1, keepdim=True)
        attn_scores = F.softmax(self.leaky_relu(attn_scores), dim=0)  # (N_R, N, K, 1)
        out = (attn_scores * node_feats).sum(dim=0)  # (N_R, N, K, d_node)
        out = self.dropout(out.view(-1, self.num_heads * self.node_hidden_dim))  # (N, K*d_node)
        return out


class RHGNNLayer(nn.Module):

    def __init__(
            self, node_in_dim, node_out_dim, rel_in_dim, rel_out_dim, num_heads,
            ntypes, etypes, dropout=0.0, negative_slope=0.2, residual=True):
        """R-HGNN层

        :param node_in_dim: int 顶点输入特征维数
        :param node_out_dim: int 顶点输出特征维数
        :param rel_in_dim: int 关系输入特征维数
        :param rel_out_dim: int 关系输出特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        :param dropout: float, optional Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        :param residual: bool, optional 是否使用残差连接，默认True
        """
        super().__init__()
        # 特定关系的卷积的参数
        fc_node = {
            ntype: nn.Linear(node_in_dim, num_heads * node_out_dim, bias=False)
            for ntype in ntypes
        }
        fc_rel = {
            etype: nn.Linear(rel_in_dim, 2 * num_heads * node_out_dim, bias=False)
            for _, etype, _ in etypes
        }
        self.rel_graph_conv = nn.ModuleDict({
            etype: RelationGraphConv(
                node_out_dim, num_heads, fc_node[stype], fc_node[dtype], fc_rel[etype],
                dropout, negative_slope, F.relu
            ) for stype, etype, dtype in etypes
        })

        # 残差连接的参数
        self.residual = residual
        if residual:
            self.fc_res = nn.ModuleDict({
                ntype: nn.Linear(node_in_dim, num_heads * node_out_dim) for ntype in ntypes
            })
            self.res_weight = nn.ParameterDict({
                ntype: nn.Parameter(torch.rand(1)) for ntype in ntypes
            })

        # 关系表示学习的参数
        self.fc_upd = nn.ModuleDict({
            etype: nn.Linear(rel_in_dim, num_heads * rel_out_dim)
            for _, etype, _ in etypes
        })

        # 跨关系消息传递的参数
        rel_attn = {
            etype: nn.Parameter(torch.FloatTensor(num_heads, node_out_dim))
            for _, etype, _ in etypes
        }
        self.rel_cross = nn.ModuleDict({
            etype: RelationCrossing(
                node_out_dim, num_heads, rel_attn[etype], dropout, negative_slope
            ) for _, etype, _ in etypes
        })

        self.rev_etype = {
            e: next(re for rs, re, rd in etypes if rs == d and rd == s)
            for s, e, d in etypes
        }
        self.reset_parameters(rel_attn)

    def reset_parameters(self, rel_attn):
        gain = nn.init.calculate_gain('relu')
        for etype in rel_attn:
            nn.init.xavier_normal_(rel_attn[etype], gain=gain)

    def forward(self, g, feats, rel_feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[(str, str, str), tensor(N_i, d_in)] 关系（三元组）到目标顶点输入特征的映射
        :param rel_feats: Dict[str, tensor(d_in_rel)] 边类型到输入关系特征的映射
        :return: Dict[(str, str, str), tensor(N_i, K*d_out)], Dict[str, tensor(K*d_out_rel)]
         关系（三元组）到目标顶点在该关系下的表示的映射、边类型到关系表示的映射
        """
        if g.is_block:
            feats_dst = {r: feats[r][:g.num_dst_nodes(r[2])] for r in feats}
        else:
            feats_dst = feats

        node_rel_feats = {
            (stype, etype, dtype): self.rel_graph_conv[etype](
                g[stype, etype, dtype],
                (feats[(dtype, self.rev_etype[etype], stype)], feats_dst[(stype, etype, dtype)]),
                rel_feats[etype]
            ) for stype, etype, dtype in g.canonical_etypes
            if g.num_edges((stype, etype, dtype)) > 0
        }  # {rel: tensor(N_dst, K*d_out)}

        if self.residual:
            for stype, etype, dtype in node_rel_feats:
                alpha = torch.sigmoid(self.res_weight[dtype])
                inherit_feat = self.fc_res[dtype](feats_dst[(stype, etype, dtype)])
                node_rel_feats[(stype, etype, dtype)] = \
                    alpha * node_rel_feats[(stype, etype, dtype)] + (1 - alpha) * inherit_feat

        out_feats = {}  # {rel: tensor(N_dst, K*d_out)}
        for stype, etype, dtype in node_rel_feats:
            dst_node_rel_feats = torch.stack([
                node_rel_feats[r] for r in node_rel_feats if r[2] == dtype
            ], dim=0)  # (N_Ri, N_i, K*d_out)
            out_feats[(stype, etype, dtype)] = self.rel_cross[etype](dst_node_rel_feats)

        rel_feats = {etype: self.fc_upd[etype](rel_feats[etype]) for etype in rel_feats}
        return out_feats, rel_feats


class RHGNN(nn.Module):

    def __init__(
            self, in_dims, hidden_dim, out_dim, rel_in_dim, rel_hidden_dim, num_heads, ntypes,
            etypes, predict_ntype, num_layers, dropout=0.0, negative_slope=0.2, residual=True):
        """R-HGNN模型

        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 顶点隐含特征维数
        :param out_dim: int 顶点输出特征维数
        :param rel_in_dim: int 关系输入特征维数
        :param rel_hidden_dim: int 关系隐含特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        :param predict_ntype: str 待预测顶点类型
        :param num_layers: int 层数
        :param dropout: float, optional Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        :param residual: bool, optional 是否使用残差连接，默认True
        """
        super().__init__()
        self._d = num_heads * hidden_dim
        self.etypes = etypes
        self.predict_ntype = predict_ntype
        # 对齐输入特征维数
        self.fc_in = nn.ModuleDict({
            ntype: nn.Linear(in_dim, num_heads * hidden_dim) for ntype, in_dim in in_dims.items()
        })
        # 关系输入特征
        self.rel_embed = nn.ParameterDict({
            etype: nn.Parameter(torch.FloatTensor(1, rel_in_dim)) for _, etype, _ in etypes
        })

        self.layers = nn.ModuleList()
        self.layers.append(RHGNNLayer(
            num_heads * hidden_dim, hidden_dim, rel_in_dim, rel_hidden_dim,
            num_heads, ntypes, etypes, dropout, negative_slope, residual
        ))
        for _ in range(1, num_layers):
            self.layers.append(RHGNNLayer(
                num_heads * hidden_dim, hidden_dim, num_heads * rel_hidden_dim, rel_hidden_dim,
                num_heads, ntypes, etypes, dropout, negative_slope, residual
            ))

        w_node = {
            etype: nn.Parameter(torch.FloatTensor(num_heads, hidden_dim, hidden_dim))
            for _, etype, _ in etypes
        }
        w_rel = {
            etype: nn.Parameter(torch.FloatTensor(num_heads, rel_hidden_dim, hidden_dim))
            for _, etype, _ in etypes
        }
        self.rel_fusing = nn.ModuleDict({
            ntype: RelationFusing(
                hidden_dim, rel_hidden_dim, num_heads,
                {e: w_node[e] for _, e, d in etypes if d == ntype},
                {e: w_rel[e] for _, e, d in etypes if d == ntype},
                dropout, negative_slope
            ) for ntype in ntypes
        })
        self.classifier = nn.Linear(num_heads * hidden_dim, out_dim)
        self.reset_parameters(self.rel_embed, w_node, w_rel)

    def reset_parameters(self, rel_embed, w_node, w_rel):
        gain = nn.init.calculate_gain('relu')
        for etype in rel_embed:
            nn.init.xavier_normal_(rel_embed[etype], gain=gain)
            nn.init.xavier_normal_(w_node[etype], gain=gain)
            nn.init.xavier_normal_(w_rel[etype], gain=gain)

    def forward(self, blocks, feats):
        """
        :param blocks: blocks: List[DGLBlock]
        :param feats: Dict[str, tensor(N_i, d_in_i)] 顶点类型到输入顶点特征的映射
        :return: tensor(N_i, d_out) 待预测顶点的最终嵌入
        """
        feats = {
            (stype, etype, dtype): self.fc_in[dtype](feats[dtype])
            for stype, etype, dtype in self.etypes
        }
        rel_feats = {rel: emb.flatten() for rel, emb in self.rel_embed.items()}
        for block, layer in zip(blocks, self.layers):
            # {(stype, etype, dtype): tensor(N_i, K*d_hid)}, {etype: tensor(K*d_hid_rel)}
            feats, rel_feats = layer(block, feats, rel_feats)

        out_feats = {
            ntype: self.rel_fusing[ntype](
                {e: feats[(s, e, d)] for s, e, d in feats if d == ntype},
                {e: rel_feats[e] for s, e, d in feats if d == ntype}
            ) for ntype in set(d for _, _, d in feats)
        }  # {ntype: tensor(N_i, K*d_hid)}
        return self.classifier(out_feats[self.predict_ntype])

    @torch.no_grad()
    def inference(self, g, feats, device, batch_size):
        """离线推断所有顶点的最终嵌入（不使用邻居采样）

        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in_i)] 顶点类型到输入顶点特征的映射
        :param device: torch.device
        :param batch_size: int 批大小
        :return: tensor(N_i, d_out) 待预测顶点的最终嵌入
        """
        feats = {
            (stype, etype, dtype): self.fc_in[dtype](feats[dtype].to(device))
            for stype, etype, dtype in self.etypes
        }
        rel_feats = {rel: emb.flatten() for rel, emb in self.rel_embed.items()}
        for layer in self.layers:
            # TODO 内存占用过大
            embeds = {
                (stype, etype, dtype): torch.zeros(g.num_nodes(dtype), self._d)
                for stype, etype, dtype in g.canonical_etypes
            }
            sampler = MultiLayerFullNeighborSampler(1)
            loader = NodeDataLoader(
                g, {ntype: torch.arange(g.num_nodes(ntype)) for ntype in g.ntypes}, sampler,
                batch_size=batch_size, shuffle=True
            )
            for input_nodes, output_nodes, blocks in tqdm(loader):
                block = blocks[0].to(device)
                in_feats = {
                    (s, e, d): feats[(s, e, d)][input_nodes[d]].to(device)
                    for s, e, d in feats
                }
                h, rel_embeds = layer(block, in_feats, rel_feats)
                for s, e, d in h:
                    embeds[(s, e, d)][output_nodes[d]] = h[(s, e, d)].cpu()
            feats = embeds
            rel_feats = rel_embeds
        feats = {r: feat.to(device) for r, feat in feats.items()}

        out_feats = {ntype: torch.zeros(g.num_nodes(ntype), self._d) for ntype in g.ntypes}
        for ntype in set(d for _, _, d in feats):
            dst_feats = {e: feats[(s, e, d)] for s, e, d in feats if d == ntype}
            dst_rel_feats = {e: rel_feats[e] for s, e, d in feats if d == ntype}
            for batch in DataLoader(torch.arange(g.num_nodes(ntype)), batch_size=batch_size):
                out_feats[ntype][batch] = self.rel_fusing[ntype](
                    {e: dst_feats[e][batch] for e in dst_rel_feats}, dst_rel_feats
                )
        return self.classifier(out_feats[self.predict_ntype])
