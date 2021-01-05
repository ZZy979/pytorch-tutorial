import random

import numpy as np
import torch
from dgl.data import citation_graph, rdf, knowledge_graph


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_citation_dataset(name):
    m = {
        'cora': citation_graph.CoraGraphDataset,
        'citeseer': citation_graph.CiteseerGraphDataset,
        'pubmed': citation_graph.PubmedGraphDataset
    }
    try:
        return m[name]()
    except KeyError:
        raise ValueError('Unknown citation dataset: {}'.format(name))


def load_rdf_dataset(name):
    m = {
        'aifb': rdf.AIFBDataset,
        'mutag': rdf.MUTAGDataset,
        'bgs': rdf.BGSDataset,
        'am': rdf.AMDataset
    }
    try:
        return m[name]()
    except KeyError:
        raise ValueError('Unknown RDF dataset: {}'.format(name))


def load_kg_dataset(name):
    m = {
        'wn18': knowledge_graph.WN18Dataset,
        'FB15k': knowledge_graph.FB15kDataset,
        'FB15k-237': knowledge_graph.FB15k237Dataset
    }
    try:
        return m[name]()
    except KeyError:
        raise ValueError('Unknown knowledge graph dataset: {}'.format(name))


def accuracy(logits, labels):
    """计算准确率

    :param logits: tensor(N, C) 预测概率，N为样本数，C为类别数
    :param labels: tensor(N) 正确标签
    :return: float 准确率
    """
    return torch.sum(torch.argmax(logits, dim=1) == labels).item() * 1.0 / len(labels)


def mean_reciprocal_rank(predicts, answers):
    """计算平均倒数排名(MRR) = sum(1 / rank_i) / N，如果答案不在预测结果中，则排名按∞计算

    例如，predicts=[[2, 0, 1], [2, 1, 0], [1, 0, 2]], answers=[1, 2, 8],
    ranks=[3, 2, ∞], MRR=(1/3+1/1+0)/3=4/9

    :param predicts: tensor(N, K) 预测结果，N为样本数，K为预测结果数
    :param answers: tensor(N) 正确答案的位置
    :return: float MRR∈[0, 1]
    """
    ranks = torch.nonzero(predicts == answers.unsqueeze(1), as_tuple=True)[1] + 1
    return torch.sum(1.0 / ranks.float()).item() / len(predicts)


def hits_at(n, predicts, answers):
    """计算Hits@n = #(rank_i <= n) / N

    例如，predicts=[[2, 0, 1], [2, 1, 0], [1, 0, 2]], answers=[1, 2, 0], ranks=[3, 1, 2], Hits@2=2/3

    :param n: int 要计算答案排名前几的比例
    :param predicts: tensor(N, K) 预测结果，N为样本数，K为预测结果数
    :param answers: tensor(N) 正确答案的位置
    :return: float Hits@n∈[0, 1]
    """
    ranks = torch.nonzero(predicts == answers.unsqueeze(1), as_tuple=True)[1] + 1
    return torch.sum(ranks <= n).float().item() / len(predicts)
