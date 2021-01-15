import itertools
import os

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from dgl.data import DGLDataset
from dgl.data.utils import download, save_graphs, load_graphs, generate_mask_tensor, idx2mask
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class IMDbDataset(DGLDataset):
    """IMDb电影数据集，只有一个异构图

    统计数据
    -----
    * 顶点：4278 movie, 5257 actor, 2081 director
    * 边：12828 movie-actor, 4278 movie-director
    * 类别数：3
    * movie顶点划分：400 train, 400 valid, 3478 test

    属性
    -----
    * num_classes: 类别数
    * metapaths: 使用的元路径
    * predict_ntype: 预测顶点类型

    movie顶点属性
    -----
    * feat: tensor(4278, 1299) 剧情关键词的词袋表示
    * label: tensor(4278) 0: Action, 1: Comedy, 2: Drama
    * train_mask, val_mask, test_mask: tensor(4278)

    actor顶点属性
    -----
    * feat: tensor(5257, 1299) 关联的电影特征的平均

    director顶点属性
    -----
    * feat: tensor(2081, 1299) 关联的电影特征的平均
    """
    _url = 'https://raw.githubusercontent.com/Jhy1993/HAN/master/data/imdb/movie_metadata.csv'
    _seed = 42

    def __init__(self):
        super().__init__('imdb', self._url)

    def download(self):
        file_path = os.path.join(self.raw_dir, 'imdb.csv')
        if not os.path.exists(file_path):
            download(self.url, path=file_path)

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        self.g = graphs[0]
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes['movie'].data[k] = self.g.nodes['movie'].data[k].type(torch.bool)

    def process(self):
        self.data = pd.read_csv(os.path.join(self.raw_dir, 'imdb.csv'), encoding='utf8') \
            .dropna(axis=0, subset=['actor_1_name', 'director_name']).reset_index(drop=True)
        self.labels = self._extract_labels()
        self.movies = list(sorted(m.strip() for m in self.data['movie_title']))
        self.directors = list(sorted(set(self.data['director_name'])))
        self.actors = list(sorted(set(itertools.chain.from_iterable(
            self.data[c].dropna().to_list()
            for c in ('actor_1_name', 'actor_2_name', 'actor_3_name')
        ))))
        self.g = self._build_graph()
        self._add_ndata()

    def _extract_labels(self):
        """提取电影类型作为标签，并删除其他类型的电影。"""
        labels = np.full(len(self.data), -1)
        for i, genres in self.data['genres'].iteritems():
            for genre in genres.split('|'):
                if genre == 'Action':
                    labels[i] = 0
                    break
                elif genre == 'Comedy':
                    labels[i] = 1
                    break
                elif genre == 'Drama':
                    labels[i] = 2
                    break
        other_idx = np.where(labels == -1)[0]
        self.data = self.data.drop(other_idx).reset_index(drop=True)
        return np.delete(labels, other_idx, axis=0)

    def _build_graph(self):
        ma, md = set(), set()
        for m, row in self.data.iterrows():
            d = self.directors.index(row['director_name'])
            md.add((m, d))
            for c in ('actor_1_name', 'actor_2_name', 'actor_3_name'):
                if row[c] in self.actors:
                    a = self.actors.index(row[c])
                    ma.add((m, a))
        ma, md = list(ma), list(md)
        ma_m, ma_a = [e[0] for e in ma], [e[1] for e in ma]
        md_m, md_d = [e[0] for e in md], [e[1] for e in md]
        return dgl.heterograph({
            ('movie', 'ma', 'actor'): (ma_m, ma_a),
            ('actor', 'am', 'movie'): (ma_a, ma_m),
            ('movie', 'md', 'director'): (md_m, md_d),
            ('director', 'dm', 'movie'): (md_d, md_m)
        })

    def _add_ndata(self):
        vectorizer = CountVectorizer(min_df=5)
        features = vectorizer.fit_transform(self.data['plot_keywords'].fillna('').values)
        self.g.nodes['movie'].data['feat'] = torch.from_numpy(features.toarray()).float()
        self.g.nodes['movie'].data['label'] = torch.from_numpy(self.labels).long()

        # actor和director顶点的特征为其关联的movie顶点特征的平均
        self.g.multi_update_all({
            'ma': (fn.copy_u('feat', 'm'), fn.mean('m', 'feat')),
            'md': (fn.copy_u('feat', 'm'), fn.mean('m', 'feat'))
        }, 'sum')

        n_movies = len(self.movies)
        train_idx, val_idx = train_test_split(np.arange(n_movies), test_size=400, random_state=self._seed)
        train_idx, test_idx = train_test_split(train_idx, train_size=400, random_state=self._seed)
        self.g.nodes['movie'].data['train_mask'] = generate_mask_tensor(idx2mask(train_idx, n_movies))
        self.g.nodes['movie'].data['val_mask'] = generate_mask_tensor(idx2mask(val_idx, n_movies))
        self.g.nodes['movie'].data['test_mask'] = generate_mask_tensor(idx2mask(test_idx, n_movies))

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
        return [['ma', 'am'], ['md', 'dm']]

    @property
    def predict_ntype(self):
        return 'movie'


class IMDb5kDataset(DGLDataset):
    """HAN作者处理的ACM数据集：https://github.com/Jhy1993/HAN

    只有一个样本，包括movie顶点基于MAM和MDM两个元路径的邻居组成的同构图

    >>> data = IMDb5kDataset()
    >>> mam_g, mdm_g = data[0]

    统计数据
    -----
    * mam_g: 4780个顶点，98010条边
    * mdm_g: 4780个顶点，21018条边
    * 类别数：3
    * 划分：300 train, 300 valid, 2687 test

    顶点属性
    -----
    * feat: tensor(4780, 1232)
    * label: tensor(4780)，-1表示无标签
    * train_mask, val_mask, test_mask: tensor(4780)
    """

    def __init__(self):
        super().__init__('imdb5k', 'https://pan.baidu.com/s/199LoAr5WmL3wgx66j-qwaw')

    def download(self):
        file_path = os.path.join(self.raw_dir, 'imdb5k.mat')
        if not os.path.exists(file_path):
            raise FileNotFoundError('请手动下载文件 {} 提取码：qkec 并保存为 {}'.format(
                self.url, file_path
            ))

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), self.gs)

    def load(self):
        self.gs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        for g in self.gs:
            for k in ('train_mask', 'val_mask', 'test_mask'):
                g.ndata[k] = g.ndata[k].type(torch.bool)

    def process(self):
        data = sio.loadmat(os.path.join(self.raw_dir, 'imdb5k.mat'))
        mam_g = dgl.graph(data['MAM'].nonzero())
        mdm_g = dgl.graph(data['MDM'].nonzero())
        # mym_g = dgl.graph(data['MYM'].nonzero())
        self.gs = [mam_g, mdm_g]

        features = torch.from_numpy(data['feature']).float()
        num_nodes = features.shape[0]
        labels = torch.full((num_nodes,), -1, dtype=torch.long)
        idx, label = data['label'].nonzero()
        labels[idx] = torch.from_numpy(label)
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
