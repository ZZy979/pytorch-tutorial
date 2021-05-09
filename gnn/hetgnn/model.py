"""Heterogeneous Graph Neural Network (HetGNN)

* 论文链接：https://dl.acm.org/doi/pdf/10.1145/3292500.3330961
* 官方代码：https://github.com/chuxuzhang/KDD2019_HetGNN
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAggregation(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        """异构内容嵌入模块，针对一种顶点类型，将该类型顶点的多个输入特征编码为一个向量

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 内容嵌入维数
        """
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, feats):
        """
        :param feats: tensor(N, C, d_in) 输入特征列表，N为batch大小，C为输入特征个数
        :return: tensor(N, d_hid) 顶点的异构内容嵌入向量
        """
        out, _ = self.lstm(feats)  # (N, C, d_hid)
        return torch.mean(out, dim=1)


class NeighborAggregation(nn.Module):

    def __init__(self, emb_dim):
        """邻居聚集模块，针对一种邻居类型t，将一个顶点的所有t类型邻居的内容嵌入向量聚集为一个向量

        :param emb_dim: int 内容嵌入维数
        """
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, emb_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, embeds):
        """
        :param embeds: tensor(N, Nt, d) 邻居的内容嵌入，N为batch大小，Nt为每个顶点的邻居个数
        :return: tensor(N, d) 顶点的t类型邻居聚集嵌入
        """
        out, _ = self.lstm(embeds)  # (N, Nt, d)
        return torch.mean(out, dim=1)


class TypesCombination(nn.Module):

    def __init__(self, emb_dim):
        """类型组合模块，针对一种顶点类型，将该类型顶点的所有类型的邻居聚集嵌入组合为一个向量

        :param emb_dim: int 邻居嵌入维数
        """
        super().__init__()
        self.attn = nn.Parameter(torch.ones(1, 2 * emb_dim))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, content_embed, neighbor_embeds):
        """
        :param content_embed: tensor(N, d) 内容嵌入，N为batch大小
        :param neighbor_embeds: tensor(A, N, d) 邻居嵌入，A为邻居类型数
        :return: tensor(N, d) 最终嵌入
        """
        neighbor_embeds = torch.cat(
            [content_embed.unsqueeze(0), neighbor_embeds], dim=0
        )  # (A+1, N, d)
        cat_embeds = torch.cat(
            [content_embed.repeat(neighbor_embeds.shape[0], 1, 1), neighbor_embeds], dim=-1
        )  # (A+1, N, 2d)
        attn_scores = self.leaky_relu((self.attn * cat_embeds).sum(dim=-1, keepdim=True))
        attn_scores = F.softmax(attn_scores, dim=0)  # (A+1, N, 1)
        out = (attn_scores * neighbor_embeds).sum(dim=0)  # (N, d)
        return out


class HetGNN(nn.Module):

    def __init__(self, in_dim, hidden_dim, ntypes):
        """HetGNN模型

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param ntypes: List[str] 顶点类型列表
        """
        super().__init__()
        self.content_aggs = nn.ModuleDict({
            ntype: ContentAggregation(in_dim, hidden_dim) for ntype in ntypes
        })
        self.neighbor_aggs = nn.ModuleDict({
            ntype: NeighborAggregation(hidden_dim) for ntype in ntypes
        })
        self.combs = nn.ModuleDict({
            ntype: TypesCombination(hidden_dim) for ntype in ntypes
        })

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, C_i, d_in)] 顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_hid)] 顶点类型到输出特征的映射
        """
        with g.local_scope():
            # 1.异构内容聚集
            for ntype in g.ntypes:
                g.nodes[ntype].data['c'] = self.content_aggs[ntype](feats[ntype])  # (N_i, d_hid)

            # 2.同类型邻居聚集
            neighbor_embeds = {}
            for dt in g.ntypes:
                tmp = []
                for st in g.ntypes:
                    # dt类型所有顶点的st类型邻居个数必须全部相同
                    g.multi_update_all({f'{st}-{dt}': (fn.copy_u('c', 'm'), stack_reducer)}, 'sum')
                    tmp.append(self.neighbor_aggs[st](g.nodes[dt].data.pop('nc')))
                neighbor_embeds[dt] = torch.stack(tmp)  # (A, N_dt, d_hid)

            # 3.类型组合
            out = {
                ntype: self.combs[ntype](g.nodes[ntype].data['c'], neighbor_embeds[ntype])
                for ntype in g.ntypes
            }
            return out

    def calc_score(self, g, h):
        """计算图中每一条边的得分 s(u, v)=h(u)^T h(v)

        :param g: DGLGraph 异构图
        :param h: Dict[str, tensor(N_i, d)] 顶点类型到顶点嵌入的映射
        :return: tensor(A*E) 所有边的得分
        """
        with g.local_scope():
            g.ndata['h'] = h
            for etype in g.etypes:
                g.apply_edges(fn.u_dot_v('h', 'h', 's'), etype=etype)
            return torch.cat(list(g.edata['s'].values())).squeeze(dim=-1)  # (A*E,)


def stack_reducer(nodes):
    return {'nc': nodes.mailbox['m']}
