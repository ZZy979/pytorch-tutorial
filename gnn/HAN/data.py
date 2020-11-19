import os
import pickle

import dgl
import numpy as np
import scipy.io as sio
import torch
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import _get_dgl_url, download, generate_mask_tensor


class ACM3025Dataset(DGLDataset):
    """HAN作者处理的ACM数据集：https://github.com/Jhy1993/HAN#qa

    只有一个图，由基于PAP和PLP(PSP)两个元路径的邻居同构图组成（只有Paper顶点）

    >>> data = ACM3025Dataset()
    >>> author_g, subject_g = data[0]

    * author_g: 3025 Paper顶点，26256条边
    * subject_g: 3025 Paper顶点，2207736条边

    两个图都有以下顶点属性：

    * feat: tensor(3025, 1870)
    * label: tensor(3025,)，类别为0~2
    * train_mask, val_mask, test_mask: tensor(3025,)，True的数量分别为600, 300, 2125
    """

    def __init__(self):
        super().__init__('ACM3025', _get_dgl_url('dataset/ACM3025.pkl'))

    def download(self):
        download(self.url, path=self._raw_file)

    def save(self):
        save_graphs(self._cache_file, self.gs)

    def load(self):
        self.gs, _ = load_graphs(self._cache_file)
        # save_graphs会将bool转换成uint8
        for g in self.gs:
            for k in ('train_mask', 'val_mask', 'test_mask'):
                g.ndata[k] = g.ndata[k].type(torch.bool)

    def process(self):
        with open(self._raw_file, 'rb') as f:
            data = pickle.load(f)
        features = torch.from_numpy(data['feature'].todense()).float()  # (3025, 1870)
        labels = torch.from_numpy(data['label'].todense()).long().nonzero()[:, 1]  # (3025,)

        # Adjacency matrices for meta-path based neighbors
        # (Mufei): I verified both of them are binary adjacency matrices with self loops
        author_g = dgl.from_scipy(data['PAP'])
        subject_g = dgl.from_scipy(data['PLP'])
        self.gs = [author_g, subject_g]

        num_nodes = data['label'].shape[0]
        train_mask = generate_mask_tensor(get_binary_mask(num_nodes, data['train_idx'][0]))
        val_mask = generate_mask_tensor(get_binary_mask(num_nodes, data['val_idx'][0]))
        test_mask = generate_mask_tensor(get_binary_mask(num_nodes, data['test_idx'][0]))

        author_g.ndata['feat'] = subject_g.ndata['feat'] = features
        author_g.ndata['label'] = subject_g.ndata['label'] = labels
        author_g.ndata['train_mask'] = subject_g.ndata['train_mask'] = train_mask
        author_g.ndata['val_mask'] = subject_g.ndata['val_mask'] = val_mask
        author_g.ndata['test_mask'] = subject_g.ndata['test_mask'] = test_mask

    def has_cache(self):
        return os.path.exists(self._cache_file)

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.gs

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return 3

    @property
    def _raw_file(self):
        return os.path.join(self.raw_dir, self.name + '.pkl')

    @property
    def _cache_file(self):
        return os.path.join(self.save_dir, self.name + '.bin')


class ACMDataset(DGLDataset):
    """ACM数据集

    只有一个图

    * 顶点：17351 author, 4025 paper, 72 field
    * 边：13407 paper-author, 4025 paper-field

    paper顶点属性：

    * feat: tensor(4025, 1903)
    * label: tensor(4025,)，类别为0~2
    * train_mask, val_mask, test_mask: tensor(4025,)，True的数量分别为808, 401, 2816
    """

    def __init__(self):
        super().__init__('ACM', _get_dgl_url('dataset/ACM.mat'))

    def download(self):
        download(self.url, path=self._raw_file)

    def save(self):
        save_graphs(self._cache_file, [self.g])

    def load(self):
        graphs, _ = load_graphs(self._cache_file)
        self.g = graphs[0]
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes['paper'].data[k] = self.g.nodes['paper'].data[k].type(torch.bool)

    def process(self):
        data = sio.loadmat(self._raw_file)
        p_vs_l = data['PvsL']  # paper-field?
        p_vs_a = data['PvsA']  # paper-author
        p_vs_t = data['PvsT']  # paper-term, bag of words
        p_vs_c = data['PvsC']  # paper-conference, labels come from that

        # We assign
        # (1) KDD papers as class 0 (data mining),
        # (2) SIGMOD and VLDB papers as class 1 (database),
        # (3) SIGCOMM and MobiCOMM papers as class 2 (communication)
        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
        p_vs_l = p_vs_l[p_selected]
        p_vs_a = p_vs_a[p_selected]
        p_vs_t = p_vs_t[p_selected]
        p_vs_c = p_vs_c[p_selected]

        self.g = dgl.heterograph({
            ('paper', 'pa', 'author'): p_vs_a.nonzero(),
            ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
            ('paper', 'pf', 'field'): p_vs_l.nonzero(),
            ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
        })
        paper_features = torch.FloatTensor(p_vs_t.toarray())  # (4025, 1903)

        pc_p, pc_c = p_vs_c.nonzero()
        paper_labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            paper_labels[pc_p[pc_c == conf_id]] = label_id
        paper_labels = torch.from_numpy(paper_labels)

        float_mask = np.zeros(len(pc_p))
        for conf_id in conf_ids:
            pc_c_mask = (pc_c == conf_id)
            float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
        train_idx = np.where(float_mask <= 0.2)[0]
        val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]

        num_paper_nodes = self.g.num_nodes('paper')
        train_mask = generate_mask_tensor(get_binary_mask(num_paper_nodes, train_idx))
        val_mask = generate_mask_tensor(get_binary_mask(num_paper_nodes, val_idx))
        test_mask = generate_mask_tensor(get_binary_mask(num_paper_nodes, test_idx))

        self.g.nodes['paper'].data['feat'] = paper_features
        self.g.nodes['paper'].data['label'] = paper_labels
        self.g.nodes['paper'].data['train_mask'] = train_mask
        self.g.nodes['paper'].data['val_mask'] = val_mask
        self.g.nodes['paper'].data['test_mask'] = test_mask

    def has_cache(self):
        return os.path.exists(self._cache_file)

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.g

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return 3

    @property
    def _raw_file(self):
        return os.path.join(self.raw_dir, self.name + '.mat')

    @property
    def _cache_file(self):
        return os.path.join(self.save_dir, self.name + '.bin')


def get_binary_mask(size, indices):
    mask = np.zeros(size)
    mask[indices] = 1
    return mask
