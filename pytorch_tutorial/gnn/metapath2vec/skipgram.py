import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embed_size):
        super().__init__()
        # nn.Embedding(v, d): range(v) -> R^d
        self.center_embed = nn.Embedding(vocab_size, embed_size)
        self.neigh_embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, center, pos, neg):
        r"""给定中心词center、正样本pos和负样本neg，返回似然函数的相反数（损失）：

        .. math::
          L=-\log {\sigma(v_c \cdot v_p)}-\sum_{n \in neg}{\log {\sigma(-v_c \cdot v_n)}}

        :param center: tensor(batch_size)，中心词
        :param pos: tensor(batch_size)，正样本
        :param neg: tensor(batch_size, neg_size)，负样本
        """
        center_embed = self.center_embed(center)  # (batch_size, d)
        pos_embed = self.neigh_embed(pos)  # (batch_size, d)
        neg_embed = self.neigh_embed(neg)  # (batch_size, neg_size, d)

        pos_score = torch.sum(center_embed * pos_embed, dim=1)  # (batch_size)
        pos_score = F.logsigmoid(torch.clamp(pos_score, min=-10, max=10))

        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze()  # (batch_size, neg_size)
        neg_score = torch.sum(F.logsigmoid(torch.clamp(neg_score, min=-10, max=10)), dim=1)  # (batch_size)
        return -torch.mean(pos_score + neg_score)
