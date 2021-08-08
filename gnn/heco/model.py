"""Self-Supervised Heterogeneous Graph Neural Network with Co-Contrastive Learning (HeCo)

* 论文链接：https://arxiv.org/pdf/2105.09111
* 官方代码：https://github.com/liun-online/HeCo
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.ops import edge_softmax
from dgl.sampling import sample_neighbors


class HeCoGATConv(nn.Module):

    def __init__(self, hidden_dim, attn_drop=0.0, negative_slope=0.01, activation=None):
        """HeCo作者代码中使用的GAT

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.01
        :param activation: callable, optional 激活函数，默认为None
        """
        super().__init__()
        self.attn_l = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_r = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain)
        nn.init.xavier_normal_(self.attn_r, gain)

    def forward(self, g, feat_src, feat_dst):
        """
        :param g: DGLGraph 邻居-目标顶点二分图
        :param feat_src: tensor(N_src, d) 邻居顶点输入特征
        :param feat_dst: tensor(N_dst, d) 目标顶点输入特征
        :return: tensor(N_dst, d) 目标顶点输出特征
        """
        with g.local_scope():
            # HeCo作者代码中使用attn_drop的方式与原始GAT不同，这样是不对的，却能顶点聚类提升性能……
            attn_l = self.attn_drop(self.attn_l)
            attn_r = self.attn_drop(self.attn_r)
            el = (feat_src * attn_l).sum(dim=-1).unsqueeze(dim=-1)  # (N_src, 1)
            er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(dim=-1)  # (N_dst, 1)
            g.srcdata.update({'ft': feat_src, 'el': el})
            g.dstdata['er'] = er
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = edge_softmax(g, e)  # (E, 1)

            # 消息传递
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            ret = g.dstdata['ft']
            if self.activation:
                ret = self.activation(ret)
            return ret


class Attention(nn.Module):

    def __init__(self, hidden_dim, attn_drop):
        """语义层次的注意力

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        """
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain)
        nn.init.xavier_normal_(self.attn, gain)

    def forward(self, h):
        """
        :param h: tensor(N, M, d) 顶点基于不同元路径/类型的嵌入，N为顶点数，M为元路径/类型数
        :return: tensor(N, d) 顶点的最终嵌入
        """
        attn = self.attn_drop(self.attn)
        # (N, M, d) -> (M, d) -> (M, 1)
        w = torch.tanh(self.fc(h)).mean(dim=0).matmul(attn.t())
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((h.shape[0],) + beta.shape)  # (N, M, 1)
        z = (beta * h).sum(dim=1)  # (N, d)
        return z


class NetworkSchemaEncoder(nn.Module):

    def __init__(self, hidden_dim, attn_drop, neighbor_sizes):
        """网络结构视图编码器

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        :param neighbor_sizes: List[int] 各邻居类型的采样个数，长度为邻居类型数S
        """
        super().__init__()
        self.gats = nn.ModuleList([
            HeCoGATConv(hidden_dim, attn_drop, activation=F.elu)
            for _ in range(len(neighbor_sizes))
        ])
        self.attn = Attention(hidden_dim, attn_drop)
        self.neighbor_sizes = neighbor_sizes

    def forward(self, bgs, feats):
        """
        :param bgs: List[DGLGraph] 各类型邻居到目标顶点的二分图
        :param feats: List[tensor(N_i, d)] 输入顶点特征，feats[0]为目标顶点特征，feats[i]对应bgs[i-1]
        :return: tensor(N_i, d) 目标顶点的最终嵌入
        """
        h = []
        for i in range(len(self.neighbor_sizes)):
            nodes = {bgs[i].dsttypes[0]: bgs[i].dstnodes()}
            sg = sample_neighbors(bgs[i], nodes, self.neighbor_sizes[i]).to(feats[i].device)
            h.append(self.gats[i](sg, feats[i + 1], feats[0]))
        h = torch.stack(h, dim=1)  # (N, S, d)
        z_sc = self.attn(h)  # (N, d)
        return z_sc


class MetapathEncoder(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, attn_drop):
        """元路径视图编码器

        :param num_metapaths: int 元路径数量M
        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        """
        super().__init__()
        self.gcns = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim, norm='right', activation=nn.PReLU())
            for _ in range(num_metapaths)
        ])
        self.attn = Attention(hidden_dim, attn_drop)

    def forward(self, mgs, feat):
        """
        :param mgs: List[DGLGraph] 基于元路径的邻居图
        :param feat: tensor(N, d) 输入顶点特征
        :return: tensor(N, d) 输出顶点特征
        """
        h = [gcn(mg, feat) for gcn, mg in zip(self.gcns, mgs)]
        h = torch.stack(h, dim=1)  # (N, M, d)
        z_mp = self.attn(h)  # (N, d)
        return z_mp


class Contrast(nn.Module):

    def __init__(self, hidden_dim, tau, lambda_):
        """对比损失模块

        :param hidden_dim: int 隐含特征维数
        :param tau: float 温度参数
        :param lambda_: float 0~1之间，网络结构视图损失的系数（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lambda_ = lambda_
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain)

    def sim(self, x, y):
        """计算相似度矩阵

        :param x: tensor(N, d)
        :param y: tensor(N, d)
        :return: tensor(N, N) S[i, j] = exp(cos(x[i], y[j]))
        """
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        numerator = torch.mm(x, y.t())
        denominator = torch.mm(x_norm, y_norm.t())
        return torch.exp(numerator / denominator / self.tau)

    def forward(self, z_sc, z_mp, pos):
        """
        :param z_sc: tensor(N, d) 目标顶点在网络结构视图下的嵌入
        :param z_mp: tensor(N, d) 目标顶点在元路径视图下的嵌入
        :param pos: tensor(N, N) 0-1张量，每个顶点的正样本
        :return: float 对比损失
        """
        z_sc_proj = self.proj(z_sc)
        z_mp_proj = self.proj(z_mp)
        sim_sc2mp = self.sim(z_sc_proj, z_mp_proj)
        sim_mp2sc = sim_sc2mp.t()

        sim_sc2mp = sim_sc2mp / (sim_sc2mp.sum(dim=1, keepdim=True) + 1e-8)  # 不能改成/=
        loss_sc = -torch.log(torch.sum(sim_sc2mp * pos, dim=1)).mean()

        sim_mp2sc = sim_mp2sc / (sim_mp2sc.sum(dim=1, keepdim=True) + 1e-8)
        loss_mp = -torch.log(torch.sum(sim_mp2sc * pos, dim=1)).mean()
        return self.lambda_ * loss_sc + (1 - self.lambda_) * loss_mp


class HeCo(nn.Module):

    def __init__(
            self, in_dims, hidden_dim, feat_drop, attn_drop, neighbor_sizes,
            num_metapaths, tau, lambda_):
        """HeCo模型

        :param in_dims: List[int] 输入特征维数，in_dims[0]对应目标顶点
        :param hidden_dim: int 隐含特征维数
        :param feat_drop: float 输入特征dropout
        :param attn_drop: float 注意力dropout
        :param neighbor_sizes: List[int] 各邻居类型到采样个数，长度为邻居类型数S
        :param num_metapaths: int 元路径数量M
        :param tau: float 温度参数
        :param lambda_: float 0~1之间，网络结构视图损失的系数（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for in_dim in in_dims])
        self.feat_drop = nn.Dropout(feat_drop)
        self.sc_encoder = NetworkSchemaEncoder(hidden_dim, attn_drop, neighbor_sizes)
        self.mp_encoder = MetapathEncoder(num_metapaths, hidden_dim, attn_drop)
        self.contrast = Contrast(hidden_dim, tau, lambda_)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for fc in self.fcs:
            nn.init.xavier_normal_(fc.weight, gain)

    def forward(self, bgs, mgs, feats, pos):
        """
        :param bgs: List[DGLGraph] 各类型邻居到目标顶点的二分图
        :param mgs: List[DGLGraph] 基于元路径的邻居图
        :param feats: List[tensor(N_i, d_in)] 输入顶点特征，feats[0]为目标顶点特征，feats[i]对应bgs[i-1]
        :param pos: tensor(N_tgt, N_tgt) 布尔张量，每个顶点的正样本
        :return: float 对比损失
        """
        h = [F.elu(self.feat_drop(fc(feat))) for fc, feat in zip(self.fcs, feats)]
        z_sc = self.sc_encoder(bgs, h)  # (N_tgt, d_hid)
        z_mp = self.mp_encoder(mgs, h[0])  # (N_tgt, d_hid)
        loss = self.contrast(z_sc, z_mp, pos)
        return loss

    @torch.no_grad()
    def get_embeds(self, mgs, feat):
        """计算目标顶点的最终嵌入(z_mp)

        :param mgs: List[DGLGraph] 基于元路径的邻居图
        :param feat: tensor(N_tgt, d_in) 目标顶点的输入特征
        :return: tensor(N_tgt, d_hid) 目标顶点的最终嵌入
        """
        h = F.elu(self.fcs[0](feat))
        z_mp = self.mp_encoder(mgs, h)
        return z_mp
