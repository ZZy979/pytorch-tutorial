import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score


def accuracy(logits, labels):
    """计算准确率

    :param logits: tensor(N, C) 预测概率，N为样本数，C为类别数
    :param labels: tensor(N) 正确标签
    :return: float 准确率
    """
    return torch.sum(torch.argmax(logits, dim=1) == labels).item() * 1.0 / len(labels)


def micro_macro_f1_score(logits, labels):
    """计算Micro-F1和Macro-F1得分

    :param logits: tensor(N, C) 预测概率，N为样本数，C为类别数
    :param labels: tensor(N) 正确标签
    :return: float, float Micro-F1和Macro-F1得分
    """
    prediction = torch.argmax(logits, dim=1).long().numpy()
    labels = labels.numpy()
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return micro_f1, macro_f1


def nmi_ari_score(logits, labels):
    """计算NMI和ARI得分

    :param logits: tensor(N, C) 预测概率，N为样本数，C为类别数
    :param labels: tensor(N) 正确标签
    :return: float, float NMI和ARI得分
    """
    num_classes = logits.shape[1]
    prediction = KMeans(n_clusters=num_classes).fit_predict(logits.detach().numpy())
    labels = labels.numpy()
    nmi = normalized_mutual_info_score(labels, prediction)
    ari = adjusted_rand_score(labels, prediction)
    return nmi, ari


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
