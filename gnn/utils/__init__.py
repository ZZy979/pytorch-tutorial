import random

import dgl
import numpy as np
import torch

from .data import *
from .metapath import *
from .metrics import *
from .neg_sampler import *
from .random_walk import *


def set_random_seed(seed):
    """设置Python, numpy, PyTorch的随机数种子

    :param seed: int 随机数种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dgl.seed(seed)


def get_device(device):
    """返回指定的GPU设备

    :param device: int GPU编号，-1表示CPU
    :return: torch.device
    """
    return torch.device(f'cuda:{device}' if device >= 0 and torch.cuda.is_available() else 'cpu')
