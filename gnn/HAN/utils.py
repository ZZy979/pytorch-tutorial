import random

import numpy as np
import torch

from gnn.HAN.data import ACM3025Dataset, ACMDataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_data(dataset):
    if dataset == 'ACM':
        return ACM3025Dataset()
    elif dataset == 'ACMRaw':
        return ACMDataset()
    else:
        return ValueError('Unsupported dataset {}'.format(dataset))
