import random

import numpy as np
import torch
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_dataset(name):
    if name == 'cora':
        return CoraGraphDataset()
    elif name == 'citeseer':
        return CiteseerGraphDataset()
    elif name == 'pubmed':
        return PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(name))


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate_accuracy(model, labels, mask, *inputs):
    model.eval()
    with torch.no_grad():
        logits = model(*inputs)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)
