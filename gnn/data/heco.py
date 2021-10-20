import os
import shutil
import zipfile

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from dgl.data import DGLDataset
from dgl.data.utils import download, save_graphs, save_info, load_graphs, load_info, \
    generate_mask_tensor, idx2mask


class HeCoDataset(DGLDataset):
    """HeCo模型使用的数据集基类

    论文链接：https://arxiv.org/pdf/2105.09111

    类属性
    -----
    * num_classes: 类别数
    * metapaths: 使用的元路径
    * predict_ntype: 目标顶点类型
    * pos: (tensor(E_pos), tensor(E_pos)) 目标顶点正样本对，pos[1][i]是pos[0][i]的正样本

    实现类
    -----
    * ACMHeCoDataset
    * DBLPHeCoDataset
    * FreebaseHeCoDataset
    * AMinerHeCoDataset
    """

    def __init__(self, name, ntypes):
        url = 'https://codeload.github.com/liun-online/HeCo/zip/refs/heads/main'
        self._ntypes = {ntype[0]: ntype for ntype in ntypes}
        super().__init__(name + '-heco', url)

    def download(self):
        file_path = os.path.join(self.raw_dir, 'HeCo-main.zip')
        if not os.path.exists(file_path):
            download(self.url, path=file_path)
        with zipfile.ZipFile(file_path, 'r') as f:
            f.extractall(self.raw_dir)
        shutil.copytree(
            os.path.join(self.raw_dir, 'HeCo-main', 'data', self.name.split('-')[0]),
            os.path.join(self.raw_path)
        )

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])
        save_info(os.path.join(self.raw_path, self.name + '_pos.pkl'), {'pos_i': self.pos_i, 'pos_j': self.pos_j})

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        self.g = graphs[0]
        ntype = self.predict_ntype
        self._num_classes = self.g.nodes[ntype].data['label'].max().item() + 1
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes[ntype].data[k] = self.g.nodes[ntype].data[k].bool()
        info = load_info(os.path.join(self.raw_path, self.name + '_pos.pkl'))
        self.pos_i, self.pos_j = info['pos_i'], info['pos_j']

    def process(self):
        self.g = dgl.heterograph(self._read_edges())

        feats = self._read_feats()
        for ntype, feat in feats.items():
            self.g.nodes[ntype].data['feat'] = feat

        labels = torch.from_numpy(np.load(os.path.join(self.raw_path, 'labels.npy'))).long()
        self._num_classes = labels.max().item() + 1
        self.g.nodes[self.predict_ntype].data['label'] = labels

        n = self.g.num_nodes(self.predict_ntype)
        for split in ('train', 'val', 'test'):
            idx = np.load(os.path.join(self.raw_path, f'{split}_20.npy'))
            mask = generate_mask_tensor(idx2mask(idx, n))
            self.g.nodes[self.predict_ntype].data[f'{split}_mask'] = mask

        pos_i, pos_j = sp.load_npz(os.path.join(self.raw_path, 'pos.npz')).nonzero()
        self.pos_i, self.pos_j = torch.from_numpy(pos_i).long(), torch.from_numpy(pos_j).long()

    def _read_edges(self):
        edges = {}
        for file in os.listdir(self.raw_path):
            name, ext = os.path.splitext(file)
            if ext == '.txt':
                u, v = name
                e = pd.read_csv(os.path.join(self.raw_path, f'{u}{v}.txt'), sep='\t', names=[u, v])
                src = e[u].to_list()
                dst = e[v].to_list()
                edges[(self._ntypes[u], f'{u}{v}', self._ntypes[v])] = (src, dst)
                edges[(self._ntypes[v], f'{v}{u}', self._ntypes[u])] = (dst, src)
        return edges

    def _read_feats(self):
        feats = {}
        for u in self._ntypes:
            file = os.path.join(self.raw_path, f'{u}_feat.npz')
            if os.path.exists(file):
                feats[self._ntypes[u]] = torch.from_numpy(sp.load_npz(file).toarray()).float()
        return feats

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
        return self._num_classes

    @property
    def metapaths(self):
        raise NotImplementedError

    @property
    def predict_ntype(self):
        raise NotImplementedError

    @property
    def pos(self):
        return self.pos_i, self.pos_j


class ACMHeCoDataset(HeCoDataset):
    """HeCo模型使用的ACM数据集

    统计数据
    -----
    * 顶点：4019 paper, 7167 author, 60 subject
    * 边：13407 paper-author, 4019 paper-subject
    * 目标顶点类型：paper
    * 类别数：3
    * 顶点划分：60 train, 1000 valid, 1000 test

    paper顶点特征
    -----
    * feat: tensor(N_paper, 1902)
    * label: tensor(N_paper) 0~2
    * train_mask, val_mask, test_mask: tensor(N_paper)

    author顶点特征
    -----
    * feat: tensor(7167, 1902)
    """

    def __init__(self):
        super().__init__('acm', ['paper', 'author', 'subject'])

    @property
    def metapaths(self):
        return [['pa', 'ap'], ['ps', 'sp']]

    @property
    def predict_ntype(self):
        return 'paper'


class DBLPHeCoDataset(HeCoDataset):
    """HeCo模型使用的DBLP数据集

    统计数据
    -----
    * 顶点：4057 author, 14328 paper, 20 conference, 7723 term
    * 边：19645 paper-author, 14328 paper-conference, 85810 paper-term
    * 目标顶点类型：author
    * 类别数：4
    * 顶点划分：80 train, 1000 valid, 1000 test

    author顶点特征
    -----
    * feat: tensor(N_author, 334)
    * label: tensor(N_author) 0~3
    * train_mask, val_mask, test_mask: tensor(N_author)

    paper顶点特征
    -----
    * feat: tensor(14328, 4231)

    term顶点特征
    -----
    * feat: tensor(7723, 50)
    """

    def __init__(self):
        super().__init__('dblp', ['author', 'paper', 'conference', 'term'])

    def _read_feats(self):
        feats = {}
        for u in 'ap':
            file = os.path.join(self.raw_path, f'{u}_feat.npz')
            feats[self._ntypes[u]] = torch.from_numpy(sp.load_npz(file).toarray()).float()
        feats['term'] = torch.from_numpy(np.load(os.path.join(self.raw_path, 't_feat.npz'))).float()
        return feats

    @property
    def metapaths(self):
        return [['ap', 'pa'], ['ap', 'pc', 'cp', 'pa'], ['ap', 'pt', 'tp', 'pa']]

    @property
    def predict_ntype(self):
        return 'author'


class FreebaseHeCoDataset(HeCoDataset):
    """HeCo模型使用的Freebase数据集

    统计数据
    -----
    * 顶点：3492 movie, 33401 author, 2502 director, 4459 writer
    * 边：65341 movie-author, 3762 movie-director, 6414 movie-writer
    * 目标顶点类型：movie
    * 类别数：3
    * 顶点划分：60 train, 1000 valid, 1000 test

    movie顶点特征
    -----
    * label: tensor(N_movie) 0~2
    * train_mask, val_mask, test_mask: tensor(N_movie)
    """

    def __init__(self):
        super().__init__('freebase', ['movie', 'author', 'director', 'writer'])

    @property
    def metapaths(self):
        return [['ma', 'am'], ['md', 'dm'], ['mw', 'wm']]

    @property
    def predict_ntype(self):
        return 'movie'


class AMinerHeCoDataset(HeCoDataset):
    """HeCo模型使用的AMiner数据集

    统计数据
    -----
    * 顶点：6564 paper, 13329 author, 35890 reference
    * 边：18007 paper-author, 58831 paper-reference
    * 目标顶点类型：paper
    * 类别数：4
    * 顶点划分：80 train, 1000 valid, 1000 test

    movie顶点特征
    -----
    * label: tensor(N_paper) 0~3
    * train_mask, val_mask, test_mask: tensor(N_paper)
    """

    def __init__(self):
        super().__init__('aminer', ['paper', 'author', 'reference'])

    @property
    def metapaths(self):
        return [['pa', 'ap'], ['pr', 'rp']]

    @property
    def predict_ntype(self):
        return 'paper'
