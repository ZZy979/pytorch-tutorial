"""Hybrid Micro/Macro Level Convolution for Heterogeneous Graph Learning (HGConv)

* 论文链接：https://arxiv.org/pdf/2012.14722
* 官方代码：https://github.com/yule-BUAA/HGConv
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import edge_softmax


class MicroConv(nn.Module):

    def __init__(self, out_dim, num_heads, feat_drop=0.0, negative_slope=0.2, activation=None):
        """微观层次卷积

        针对一种关系（边类型）R=<stype, etype, dtype>，聚集关系R下的邻居信息，得到关系R关于dtype类型顶点的表示
        （特征转换矩阵和注意力向量是与顶点类型相关的，除此之外与GAT完全相同）

        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param feat_drop: float, optional 输入特征Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        :param activation: callable, optional 用于输出特征的激活函数，默认为None
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

    def forward(self, g, feat_src, feat_dst, fc_src, fc_dst, attn_src):
        """
        :param g: DGLGraph 二分图（只包含一种关系）
        :param feat_src: tensor(N_src, d_in) 源顶点输入特征
        :param feat_dst: tensor(N_dst, d_in) 目标顶点输入特征
        :param fc_src: nn.Linear(d_in, K*d_out) 源顶点特征转换模块
        :param fc_dst: nn.Linear(d_in, K*d_out) 目标顶点特征转换模块
        :param attn_src: nn.Parameter(K, 2d_out) 源顶点类型对应的注意力向量
        :return: tensor(N_dst, K*d_out) 该关系关于目标顶点的表示
        """
        with g.local_scope():
            feat_src = fc_src(self.feat_drop(feat_src)).view(-1, self.num_heads, self.out_dim)
            feat_dst = fc_dst(self.feat_drop(feat_dst)).view(-1, self.num_heads, self.out_dim)

            # a^T (z_u || z_v) = (a_l^T || a_r^T) (z_u || z_v) = a_l^T z_u + a_r^T z_v = el + er
            el = (feat_src * attn_src[:, :self.out_dim]).sum(dim=-1, keepdim=True)  # (N_src, K, 1)
            er = (feat_dst * attn_src[:, self.out_dim:]).sum(dim=-1, keepdim=True)  # (N_dst, K, 1)
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


class MacroConv(nn.Module):

    def __init__(self, out_dim, num_heads, dropout=0.0, negative_slope=0.2):
        """宏观层次卷积

        针对所有关系（边类型），将每种类型的顶点关联的所有关系关于该类型顶点的表示组合起来

        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param dropout: float, optional Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, node_feats, rel_feats, fc_node, fc_rel, attn):
        """
        :param node_feats: Dict[str, tensor(N_i, d_in) 顶点类型到输入顶点特征的映射
        :param rel_feats: Dict[(str, str, str), tensor(N_i, K*d_out)]
         关系(stype, etype, dtype)到关系关于其终点类型的表示的映射
        :param fc_node: Dict[str, nn.Linear(d_in, K*d_out)] 顶点类型到顶点特征转换模块的映射
        :param fc_rel: Dict[str, nn.Linear(K*d_out, K*d_out)] 关系到关系表示转换模块的映射
        :param attn: nn.Parameter(K, 2d_out)
        :return: Dict[str, tensor(N_i, K*d_out)] 顶点类型到最终顶点嵌入的映射
        """
        node_feats = {
            ntype: fc_node[ntype](feat).view(-1, self.num_heads, self.out_dim)
            for ntype, feat in node_feats.items()
        }
        rel_feats = {
            rel: fc_rel[str(rel)](feat).view(-1, self.num_heads, self.out_dim)
            for rel, feat in rel_feats.items()
        }
        out_feats = {}
        for ntype, node_feat in node_feats.items():
            rel_node_feats = [feat for rel, feat in rel_feats.items() if rel[2] == ntype]
            if len(rel_node_feats) == 1:
                out_feats[ntype] = rel_node_feats[0].view(-1, self.num_heads * self.out_dim)
            else:
                rel_node_feats = torch.stack(rel_node_feats, dim=0)  # (R, N_i, K, d_out)
                cat_feats = torch.cat(
                    (node_feat.repeat(rel_node_feats.shape[0], 1, 1, 1), rel_node_feats), dim=-1
                )  # (R, N_i, K, 2d_out)
                attn_scores = self.leaky_relu((attn * cat_feats).sum(dim=-1, keepdim=True))
                attn_scores = F.softmax(attn_scores, dim=0)  # (R, N_i, K, 1)
                out_feat = (attn_scores * rel_node_feats).sum(dim=0)  # (N_i, K, d_out)
                out_feats[ntype] = self.dropout(out_feat.reshape(-1, self.num_heads * self.out_dim))
        return out_feats


class HGConvLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, g, dropout=0.0, residual=True):
        """HGConv层

        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param g: DGLGraph 异构图
        :param dropout: float, optional Dropout概率，默认为0
        :param residual: bool, optional 是否使用残差连接，默认True
        """
        super().__init__()
        self.residual = residual

        # 微观层次卷积的参数
        self.micro_fc = nn.ModuleDict({
            ntype: nn.Linear(in_dim, num_heads * out_dim, bias=False) for ntype in g.ntypes
        })
        self.micro_attn = nn.ParameterDict({
            ntype: nn.Parameter(torch.FloatTensor(size=(num_heads, 2 * out_dim)))
            for ntype in g.ntypes
        })

        # 宏观层次卷积的参数
        self.macro_fc_node = nn.ModuleDict({
            ntype: nn.Linear(in_dim, num_heads * out_dim, bias=False) for ntype in g.ntypes
        })
        self.macro_fc_rel = nn.ModuleDict({
            # nn.ModuleDict不支持元组作为键
            str(rel): nn.Linear(num_heads * out_dim, num_heads * out_dim, bias=False)
            for rel in g.canonical_etypes
        })
        self.macro_attn = nn.Parameter(torch.FloatTensor(size=(num_heads, 2 * out_dim)))

        self.micro_conv = nn.ModuleDict({
            str(rel): MicroConv(out_dim, num_heads, dropout, activation=F.relu)
            for rel in g.canonical_etypes
        })
        self.macro_conv = MacroConv(out_dim, num_heads, dropout)
        if residual:
            self.res_fc = nn.ModuleDict({
                ntype: nn.Linear(in_dim, num_heads * out_dim) for ntype in g.ntypes
            })
            self.res_weight = nn.ParameterDict({
                ntype: nn.Parameter(torch.rand(1)) for ntype in g.ntypes
            })
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for ntype in self.micro_fc:
            nn.init.xavier_normal_(self.micro_fc[ntype].weight, gain=gain)
            nn.init.xavier_normal_(self.micro_attn[ntype], gain=gain)
            nn.init.xavier_normal_(self.macro_fc_node[ntype].weight, gain=gain)
            if self.residual:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
        for rel in self.macro_fc_rel:
            nn.init.xavier_normal_(self.macro_fc_rel[rel].weight, gain=gain)
        nn.init.xavier_normal_(self.macro_attn, gain=gain)

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入顶点特征的映射
        :return: Dict[str, tensor(N_i, K*d_out)] 顶点类型到最终顶点嵌入的映射
        """
        rel_feats = {
            (stype, etype, dtype): self.micro_conv[str((stype, etype, dtype))](
                g[stype, etype, dtype], feats[stype], feats[dtype],
                self.micro_fc[stype], self.micro_fc[dtype], self.micro_attn[stype]
            )
            for stype, etype, dtype in g.canonical_etypes
        }  # {rel: tensor(N_i, K*d_out)}
        out_feats = self.macro_conv(
            feats, rel_feats, self.macro_fc_node, self.macro_fc_rel, self.macro_attn
        )  # {ntype: tensor(N_i, K*d_out)}
        if self.residual:
            for ntype in out_feats:
                alpha = torch.sigmoid(self.res_weight[ntype])
                inherit_feat = self.res_fc[ntype](feats[ntype])
                out_feats[ntype] = alpha * out_feats[ntype] + (1 - alpha) * inherit_feat
        return out_feats


class HGConv(nn.Module):

    def __init__(
            self, in_dims, hidden_dim, out_dim, num_heads, g, predict_ntype,
            num_layers, dropout=0.0, residual=True):
        """HGConv模型

        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param g: DGLGraph 异构图
        :param predict_ntype: str 待预测顶点类型
        :param num_layers: int 层数
        :param dropout: float, optional Dropout概率，默认为0
        :param residual: bool, optional 是否使用残差连接，默认True
        """
        super().__init__()
        self.predict_ntype = predict_ntype
        # 对齐输入特征维数
        self.fc_in = nn.ModuleDict({
            ntype: nn.Linear(in_dim, num_heads * hidden_dim) for ntype, in_dim in in_dims.items()
        })
        self.layers = nn.ModuleList([
            HGConvLayer(num_heads * hidden_dim, hidden_dim, num_heads, g, dropout, residual)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(num_heads * hidden_dim, out_dim)

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in_i)] 顶点类型到输入顶点特征的映射
        :return: tensor(N_i, d_out) 待预测顶点的最终嵌入
        """
        feats = {ntype: self.fc_in[ntype](feat) for ntype, feat in feats.items()}
        for i in range(len(self.layers)):
            feats = self.layers[i](g, feats)  # {ntype: tensor(N_i, K*d_hid)}
        return self.classifier(feats[self.predict_ntype])
