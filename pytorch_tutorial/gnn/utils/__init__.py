import random

import numpy as np
import torch

from .data import load_citation_dataset, load_rdf_dataset, load_kg_dataset
from .metrics import accuracy, mean_reciprocal_rank, hits_at


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
