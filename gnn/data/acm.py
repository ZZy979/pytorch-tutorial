import os
import pickle

import dgl
import dgl.function as fn
import numpy as np
import scipy.io as sio
import torch
from dgl.data import DGLDataset
from dgl.data.utils import _get_dgl_url, download, save_graphs, load_graphs, \
    generate_mask_tensor, idx2mask


class ACMDataset(DGLDataset):
    """ACM数据集，只有一个异构图

    统计数据
    -----
    * 顶点：17351 author, 4025 paper, 72 field
    * 边：13407 paper-author, 4025 paper-field
    * 类别数：3
    * paper顶点划分：808 train, 401 valid, 2816 test

    属性
    -----
    * num_classes: 类别数
    * metapaths: 使用的元路径
    * predict_ntype: 预测顶点类型

    paper顶点属性
    -----
    * feat: tensor(4025, 1903) 关键词的词袋表示
    * label: tensor(4025)
    * train_mask, val_mask, test_mask: tensor(4025)

    author顶点属性
    -----
    * feat: tensor(17351, 1903) 关联的论文特征的平均

    field顶点属性
    -----
    * feat: tensor(72, 72) one-hot编码
    """

    def __init__(self):
        super().__init__('ACM', _get_dgl_url('dataset/ACM.mat'))

    def download(self):
        file_path = os.path.join(self.raw_dir, 'ACM.mat')
        if not os.path.exists(file_path):
            download(self.url, path=file_path)

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        self.g = graphs[0]
        # save_graphs会将bool转换成uint8
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes['paper'].data[k] = self.g.nodes['paper'].data[k].bool()

    def process(self):
        data = sio.loadmat(os.path.join(self.raw_dir, 'ACM.mat'))
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
        train_mask = generate_mask_tensor(idx2mask(train_idx, num_paper_nodes))
        val_mask = generate_mask_tensor(idx2mask(val_idx, num_paper_nodes))
        test_mask = generate_mask_tensor(idx2mask(test_idx, num_paper_nodes))

        self.g.nodes['paper'].data['feat'] = paper_features
        self.g.nodes['paper'].data['label'] = paper_labels
        self.g.nodes['paper'].data['train_mask'] = train_mask
        self.g.nodes['paper'].data['val_mask'] = val_mask
        self.g.nodes['paper'].data['test_mask'] = test_mask
        # author顶点的特征为其关联的paper顶点特征的平均
        self.g.multi_update_all({'pa': (fn.copy_u('feat', 'm'), fn.mean('m', 'feat'))}, 'sum')
        self.g.nodes['field'].data['feat'] = torch.eye(self.g.num_nodes('field'))

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))

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
    def metapaths(self):
        return [['pa', 'ap'], ['pf', 'fp']]

    @property
    def predict_ntype(self):
        return 'paper'


class ACM3025Dataset(DGLDataset):
    """HAN作者处理的ACM数据集：https://github.com/Jhy1993/HAN#qa

    只有一个样本，包括paper顶点基于PAP和PLP两个元路径的邻居组成的同构图

    >>> data = ACM3025Dataset()
    >>> author_g, subject_g = data[0]

    统计数据
    -----
    * author_g: 3025个顶点，29281条边
    * subject_g: 3025个顶点，2210761条边
    * 类别数：3
    * 划分：600 train, 300 valid, 2125 test

    顶点属性
    -----
    * feat: tensor(3025, 1870)
    * label: tensor(3025)
    * train_mask, val_mask, test_mask: tensor(3025)
    """

    def __init__(self):
        super().__init__('ACM3025', _get_dgl_url('dataset/ACM3025.pkl'))

    def download(self):
        file_path = os.path.join(self.raw_dir, 'ACM3025.pkl')
        if not os.path.exists(file_path):
            download(self.url, path=file_path)

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), self.gs)

    def load(self):
        self.gs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        for g in self.gs:
            for k in ('train_mask', 'val_mask', 'test_mask'):
                g.ndata[k] = g.ndata[k].bool()

    def process(self):
        with open(os.path.join(self.raw_dir, 'ACM3025.pkl'), 'rb') as f:
            data = pickle.load(f)
        features = torch.from_numpy(data['feature'].todense()).float()  # (3025, 1870)
        labels = torch.from_numpy(data['label'].todense()).long().nonzero(as_tuple=True)[1]  # (3025)

        # Adjacency matrices for meta-path based neighbors
        # (Mufei): I verified both of them are binary adjacency matrices with self loops
        author_g = dgl.from_scipy(data['PAP'])
        subject_g = dgl.from_scipy(data['PLP'])
        self.gs = [author_g, subject_g]

        num_nodes = data['label'].shape[0]
        train_mask = generate_mask_tensor(idx2mask(data['train_idx'][0], num_nodes))
        val_mask = generate_mask_tensor(idx2mask(data['val_idx'][0], num_nodes))
        test_mask = generate_mask_tensor(idx2mask(data['test_idx'][0], num_nodes))
        for g in self.gs:
            g.ndata['feat'] = features
            g.ndata['label'] = labels
            g.ndata['train_mask'] = train_mask
            g.ndata['val_mask'] = val_mask
            g.ndata['test_mask'] = test_mask

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.gs

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return 3
