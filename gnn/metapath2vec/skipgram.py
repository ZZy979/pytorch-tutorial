import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        self.neigh_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, pos, neg):
        r"""给定中心词、正样本和负样本，返回似然函数的相反数（损失）：

        .. math::
          L=-\log {\sigma(v_c \cdot v_p)}-\sum_{n \in neg}{\log {\sigma(-v_c \cdot v_n)}}

        :param center: tensor(N) 中心词
        :param pos: tensor(N) 正样本
        :param neg: tensor(N, M) 负样本
        """
        center_embed = self.center_embed(center)  # (N, d)
        pos_embed = self.neigh_embed(pos)  # (N, d)
        neg_embed = self.neigh_embed(neg)  # (N, M, d)

        pos_score = torch.sum(center_embed * pos_embed, dim=1)  # (N)
        pos_score = F.logsigmoid(torch.clamp(pos_score, min=-10, max=10))

        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze()  # (N, M)
        neg_score = torch.sum(F.logsigmoid(torch.clamp(neg_score, min=-10, max=10)), dim=1)  # (N)
        return -torch.mean(pos_score + neg_score)
