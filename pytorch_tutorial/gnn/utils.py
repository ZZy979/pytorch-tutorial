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
    return torch.sum(torch.argmax(logits, dim=1) == labels).item() * 1.0 / len(labels)
