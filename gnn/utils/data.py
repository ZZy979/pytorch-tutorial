import dgl
from dgl.data import citation_graph, rdf, knowledge_graph
from dgl.utils import extract_node_subframes, set_new_frames
from sklearn.model_selection import train_test_split

__all__ = [
    'load_citation_dataset', 'load_rdf_dataset', 'load_kg_dataset',
    'split_idx', 'add_reverse_edges'
]


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


def split_idx(samples, train_size, val_size, random_state=None):
    """将samples划分为训练集、测试集和验证集，需满足（用浮点数表示）：

    * 0 < train_size < 1
    * 0 < val_size < 1
    * train_size + val_size < 1

    :param samples: list/ndarray/tensor 样本集
    :param train_size: int or float 如果是整数则表示训练样本的绝对个数，否则表示训练样本占所有样本的比例
    :param val_size: int or float 如果是整数则表示验证样本的绝对个数，否则表示验证样本占所有样本的比例
    :param random_state: int, optional 随机数种子
    :return: (train, val, test) 类型与samples相同
    """
    train, val = train_test_split(samples, train_size=train_size, random_state=random_state)
    if isinstance(val_size, float):
        val_size *= len(samples) / len(val)
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test


def add_reverse_edges(g):
    """给异构图的每种边添加反向边，返回新的异构图

    :param g: DGLGraph 异构图
    :return: DGLGraph 添加反向边之后的异构图
    """
    data = {}
    for stype, etype, dtype in g.canonical_etypes:
        u, v = g.edges(etype=(stype, etype, dtype))
        data[(stype, etype, dtype)] = u, v
        data[(dtype, etype + '_rev', stype)] = v, u
    new_g = dgl.heterograph(data, {ntype: g.num_nodes(ntype) for ntype in g.ntypes})
    node_frames = extract_node_subframes(g, None)
    set_new_frames(new_g, node_frames=node_frames)
    return new_g
