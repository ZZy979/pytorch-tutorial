"""Knowledge Graph Convolutional Networks for Recommender Systems (KGCN)

论文链接：https://arxiv.org/pdf/1904.12575
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class KGCNLayer(nn.Module):

    def __init__(self, hidden_dim, aggregator):
        """KGCN层

        :param hidden_dim: int 隐含特征维数d
        :param aggregator: str 实体表示与邻居表示的组合方式：sum, concat, neighbor
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregator = aggregator
        if aggregator == 'concat':
            self.w = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.w = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, src_feat, dst_feat, rel_feat, user_feat, activation):
        """
        :param src_feat: tensor(B, K^h, K, d) 输入实体表示，B为batch大小，K为邻居个数，h为跳步数/层数
        :param dst_feat: tensor(B, K^h, d) 目标实体表示
        :param rel_feat: tensor(B, K^h, K, d) 关系表示
        :param user_feat: tensor(B, d) 用户表示
        :param activation: callable 激活函数
        :return: tensor(B, K^h, d) 输出实体表示
        """
        batch_size = user_feat.shape[0]
        user_feat = user_feat.view(batch_size, 1, 1, self.hidden_dim)  # (B, 1, 1, d)

        user_rel_scores = (user_feat * rel_feat).sum(dim=-1)  # (B, K^h, K)
        user_rel_scores = F.softmax(user_rel_scores, dim=-1).unsqueeze(dim=-1)  # (B, K^h, K, 1)
        agg = (user_rel_scores * src_feat).sum(dim=2)  # (B, K^h, d)

        if self.aggregator == 'sum':
            out = (dst_feat + agg).view(-1, self.hidden_dim)  # (B*K^h, d)
        elif self.aggregator == 'concat':
            out = torch.cat([dst_feat, agg], dim=-1).view(-1, 2 * self.hidden_dim)  # (B*K^h, 2d)
        else:
            out = agg.view(-1, self.hidden_dim)  # (B*K^h, d)

        out = self.w(out).view(batch_size, -1, self.hidden_dim)  # (B, K^h, d)
        return activation(out)


class KGCN(nn.Module):

    def __init__(self, hidden_dim, neighbor_size, aggregator, num_hops, num_users, num_entities, num_rels):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.neighbor_size = neighbor_size
        self.num_hops = num_hops
        self.aggregator = KGCNLayer(hidden_dim, aggregator)
        self.user_embed = nn.Embedding(num_users, hidden_dim)
        self.entity_embed = nn.Embedding(num_entities, hidden_dim)
        self.rel_embed = nn.Embedding(num_rels, hidden_dim)

    def forward(self, pair_graph, blocks):
        """
        :param pair_graph: DGLGraph 用户-物品子图
        :param blocks: List[DGLBlock] 知识图谱的MFG，blocks[-1].dstnodes()对应items
        :return: tensor(B) 用户-物品预测概率
        """
        u, v = pair_graph.edges()
        users = pair_graph.nodes['user'].data[dgl.NID][u]
        user_feat = self.user_embed(users)  # (B, d)
        entities, relations = self._get_neighbors(v, blocks)
        item_feat = self._aggregate(entities, relations, user_feat)  # (B, d)
        scores = (user_feat * item_feat).sum(dim=-1)
        return torch.sigmoid(scores)

    def _get_neighbors(self, v, blocks):
        batch_size = v.shape[0]
        entities, relations = [blocks[-1].dstdata[dgl.NID][v].unsqueeze(dim=-1)], []
        for b in reversed(blocks):
            u, dst = b.in_edges(v)
            entities.append(b.srcdata[dgl.NID][u].view(batch_size, -1))
            relations.append(b.edata['relation'][b.edge_ids(u, dst)].view(batch_size, -1))
            v = u
        return entities, relations

    def _aggregate(self, entities, relations, user_feat):
        batch_size = user_feat.shape[0]
        entity_feats = [self.entity_embed(entity) for entity in entities]
        rel_feats = [self.rel_embed(rel) for rel in relations]
        for h in range(self.num_hops):
            activation = torch.tanh if h == self.num_hops - 1 else torch.sigmoid
            new_entity_feats = [
                self.aggregator(
                    entity_feats[i + 1].view(batch_size, -1, self.neighbor_size, self.hidden_dim),
                    entity_feats[i],
                    rel_feats[i].view(batch_size, -1, self.neighbor_size, self.hidden_dim),
                    user_feat, activation
                ) for i in range(self.num_hops - h)
            ]
            entity_feats = new_entity_feats
        return entity_feats[0].view(batch_size, self.hidden_dim)
