from pytorch_tutorial.gnn.data import ACMDataset, ACM3025Dataset, DBLPFourAreaDataset, DBLP4057Dataset


def load_hetero_data(dataset):
    """原始异构图数据集，由一个异构图组成。"""
    if dataset == 'ACM':
        data = ACMDataset()
        ntype = 'paper'
    elif dataset == 'DBLP':
        data = DBLPFourAreaDataset()
        ntype = 'author'
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))

    g = data[0]
    ndata = g.nodes[ntype].data
    return g, data.meta_paths, ndata['feat'], ndata['label'], data.num_classes, \
           ndata['train_mask'], ndata['val_mask'], ndata['test_mask']


def load_data(dataset):
    """处理过的异构图数据集，由基于多条元路径的邻居生成的同构图组成。"""
    if dataset == 'ACM':
        data = ACM3025Dataset()
    elif dataset == 'DBLP':
        data = DBLP4057Dataset()
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))

    gs = data[0]
    g = gs[0]
    return gs, g.ndata['feat'], g.ndata['label'], data.num_classes, \
           g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
