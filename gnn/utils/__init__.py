import numpy as np

from .data import *
from .metapath import *
from .metrics import *
from .neg_sampler import *
from .random_walk import metapath_random_walk


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
