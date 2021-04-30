import os

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import makedirs, download, save_graphs, load_graphs


class AMiner2Dataset(DGLDataset):
    """HetGNN作者使用的Academic II学术网络数据集

    https://dl.acm.org/doi/pdf/10.1145/3292500.3330961

    统计数据
    -----
    * 顶点：28646 author, 21044 paper, 18 venue
    * 边：69311 author-paper, 34238 paper-paper, 21044 paper-venue
    * 类别数：4

    属性
    -----
    * num_classes: 类别数

    author顶点属性
    -----
    * label: tensor(28646) -1~3，-1表示无标签

    paper顶点属性
    -----
    * title_embed: tensor(21044, 128) 预训练的标题词向量
    * abstract_embed: tensor(21044, 128) 预训练的摘要词向量
    * year: tensor(21044) 1~10，论文年份-2005（即2006~2015）
    """
    _url = 'https://raw.githubusercontent.com/chuxuzhang/KDD2019_HetGNN/master/data/academic/'
    _raw_files = [
        'a_p_list_train.txt', 'a_p_list_test.txt', 'p_v.txt',
        'p_p_cite_list_train.txt', 'p_p_cite_list_test.txt',
        'a_class_train.txt', 'a_class_test.txt',
        'p_title_embed.txt', 'p_abstract_embed.txt', 'p_time.txt'
    ]

    def __init__(self):
        super().__init__('aminer2', self._url)

    def download(self):
        if not os.path.exists(self.raw_path):
            makedirs(self.raw_path)
        for file in self._raw_files:
            download(self.url + file, os.path.join(self.raw_path, file))

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        self.g = graphs[0]

    def process(self):
        author_paper, paper_paper, paper_venue, author_label, \
            paper_title_embed, paper_abstract_embed, paper_time = self._read_raw_data()

        ap_a, ap_p = author_paper['u'].to_list(), author_paper['v'].to_list()
        pp_u, pp_v = paper_paper['u'].to_list(), paper_paper['v'].to_list()
        pv_p, pv_v = paper_venue['u'].to_list(), paper_venue['v'].to_list()
        g = dgl.heterograph({
            ('author', 'write', 'paper'): (ap_a, ap_p),
            ('paper', 'write_rev', 'author'): (ap_p, ap_a),
            ('paper', 'cite', 'paper'): (pp_u, pp_v),
            ('paper', 'cite_rev', 'paper'): (pp_v, pp_u),
            ('paper', 'publish', 'venue'): (pv_p, pv_v),
            ('venue', 'publish_rev', 'paper'): (pv_v, pv_p)
        })
        g.nodes['paper'].data.update({
            'title_embed': torch.from_numpy(paper_title_embed),
            'abstract_embed': torch.from_numpy(paper_abstract_embed),
            'year': torch.from_numpy(paper_time['year'].to_numpy())
        })
        g.nodes['author'].data['label'] = torch.from_numpy(author_label['label'].to_numpy())
        self.g = g

    def _read_raw_data(self):
        author_paper = self._read_edge_file('a_p_list_{}.txt')
        paper_paper = self._read_edge_file('p_p_cite_list_{}.txt')
        paper_venue = pd.read_csv(os.path.join(self.raw_path, 'p_v.txt'), names=['u', 'v'])
        author_label = self._read_author_label(author_paper['u'].max() + 1)
        paper_title_embed = self._read_embed_file('p_title_embed.txt')
        paper_abstract_embed = self._read_embed_file('p_abstract_embed.txt')
        paper_time = pd.read_csv(
            os.path.join(self.raw_path, 'p_time.txt'), sep='\t', names=['pid', 'year']
        )
        return author_paper, paper_paper, paper_venue, author_label, \
               paper_title_embed, paper_abstract_embed, paper_time

    def _read_edge_file(self, filename):
        # uid:vid,vid,...
        edges = []
        for t in ('train', 'test'):
            src, dst = [], []
            with open(os.path.join(self.raw_path, filename.format(t))) as f:
                for line in f:
                    u, vs = line.strip().split(':')
                    u = int(u)
                    vs = [int(v) for v in vs.split(',')]
                    for v in vs:
                        src.append(u)
                        dst.append(v)
                edges.append(pd.DataFrame({'u': src, 'v': dst, 'train': int(t == 'train')}))
        return pd.concat(edges, ignore_index=True)

    def _read_author_label(self, num_authors):
        author_label = pd.concat([
            pd.read_csv(os.path.join(self.raw_path, f'a_class_{t}.txt'), names=['aid', 'label'])
            for t in ('train', 'test')
        ])
        author_label = author_label.append(pd.DataFrame({
            'aid': list(set(range(num_authors)) - set(author_label['aid'].tolist())),
            'label': -1
        }))
        author_label.sort_values('aid', inplace=True)
        return author_label

    def _read_embed_file(self, filename):
        embed = []
        with open(os.path.join(self.raw_path, filename)) as f:
            f.readline()
            for line in f:
                embed.append(np.asfarray(line.strip().split(' ', 1)[1].split(' '), 'float32'))
        return np.array(embed)

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
        return 4
