import random

import torch
from dgl.dataloading.negative_sampler import _BaseNegativeSampler


class RatioNegativeSampler(_BaseNegativeSampler):

    def __init__(self, neg_sample_ratio=1.0):
        """按一定正样本边的比例采样负样本边的负采样器

        :param neg_sample_ratio: float, optional 负样本边数量占正样本边数量的（大致）比例，默认为1
        """
        self.neg_sample_ratio = neg_sample_ratio

    def _generate(self, g, eids, canonical_etype):
        stype, _, dtype = canonical_etype
        num_src_nodes, num_dst_nodes = g.num_nodes(stype), g.num_nodes(dtype)
        total = num_src_nodes * num_dst_nodes

        # 处理|Vs×Vd-E|<r|E|的情况
        num_neg_samples = min(int(self.neg_sample_ratio * len(eids)), total - len(eids))
        # 为了去重需要多采样一部分
        alpha = abs(1 / (1 - 1.1 * len(eids) / total))

        # 将边转换为0~|Vs||Vd|-1的编号
        src, dst = g.find_edges(eids, etype=canonical_etype)
        idx = set((src * num_dst_nodes + dst).tolist())

        neg_idx = set(random.sample(range(total), min(int(alpha * num_neg_samples), total))) - idx
        neg_idx = torch.tensor(list(neg_idx))[:num_neg_samples]
        return neg_idx // num_dst_nodes, neg_idx % num_dst_nodes
