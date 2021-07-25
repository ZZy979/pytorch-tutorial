import csv
import os
import zipfile
from os.path import join

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import makedirs, download, save_graphs, save_info, load_graphs, load_info


class AMinerCSDataset(DGLDataset):
    """AMiner计算机科学领域学术网络数据集，只有一个异构图

    来自论文metapath2vec: https://ericdongyx.github.io/metapath2vec/m2v.html

    统计数据
    -----
    * 顶点：1693531 author, 3194405 paper, 3883 conf
    * 边：9323739 paper-author, 3194405 paper-conf
    * 类别数：8

    属性
    -----
    * num_classes: int 类别数
    * author_names: List[str] 学者姓名
    * paper_titles: List[str] 论文标题
    * conf_names: List[str] 期刊名称

    author顶点属性
    -----
    * label: tensor(1693531) -1~7, -1表示无标签

    conf顶点属性
    -----
    * label: tensor(3883) -1~7, -1表示无标签
    """
    _data_url = 'https://www.dropbox.com/s/1bnz8r7mofx0osf/net_aminer.zip?dl=1'
    _label_url = 'https://www.dropbox.com/s/nkocx16rpl4ydde/label.zip?dl=1'

    def __init__(self):
        super().__init__('aminer-cs', self._data_url)

    def download(self):
        self._download_and_extract(self._data_url, 'net_aminer.zip')
        self._download_and_extract(self._label_url, 'net_aminer_label.zip')

    def _download_and_extract(self, url, filename):
        file_path = join(self.raw_dir, filename)
        if not os.path.exists(file_path):
            download(url, file_path)
        with zipfile.ZipFile(file_path, 'r') as f:
            f.extractall(self.raw_path)

    def save(self):
        save_graphs(join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])
        save_info(join(self.save_path, self.name + '_info.pkl'), {
            'author_names': self.author_names,
            'paper_titles': self.paper_titles,
            'conf_names': self.conf_names
        })

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        self.g = graphs[0]
        info = load_info(join(self.save_path, self.name + '_info.pkl'))
        self.author_names = info['author_names']
        self.paper_titles = info['paper_titles']
        self.conf_names = info['conf_names']

    def process(self):
        authors, papers, confs, paper_author, paper_conf = self._read_raw_data()

        pa_p, pa_a = paper_author['pid'].to_list(), paper_author['aid'].to_list()
        pc_p, pc_c = paper_conf['pid'].to_list(), paper_conf['cid'].to_list()
        self.g = dgl.heterograph({
            ('paper', 'pa', 'author'): (pa_p, pa_a),
            ('author', 'ap', 'paper'): (pa_a, pa_p),
            ('paper', 'pc', 'conf'): (pc_p, pc_c),
            ('conf', 'cp', 'paper'): (pc_c, pc_p)
        })
        self.g.nodes['author'].data['label'] = torch.from_numpy(authors['label'].to_numpy())
        self.g.nodes['conf'].data['label'] = torch.from_numpy(confs['label'].to_numpy())

        self.author_names = authors['name'].tolist()
        self.paper_titles = papers['title'].tolist()
        self.conf_names = confs['name'].tolist()

    def _read_raw_data(self):
        authors = self._read_file(join('net_aminer', 'id_author.txt'), '\t', ['id', 'name'], 'id')
        authors.sort_index(inplace=True)
        papers = self._read_file(join('net_aminer', 'paper.txt'), '\t', ['id', 'title'], 'id')
        papers.sort_index(inplace=True)
        confs = self._read_file(join('net_aminer', 'id_conf.txt'), '\t', ['id', 'name'], 'id')
        confs.sort_index(inplace=True)
        paper_author = self._read_file(join('net_aminer', 'paper_author.txt'), '\t', ['pid', 'aid'])
        paper_conf = self._read_file(join('net_aminer', 'paper_conf.txt'), '\t', ['pid', 'cid'])

        author_label = self._read_file(
            join('label', 'googlescholar.8area.author.label.txt'), ' ', ['name', 'label'], 'name'
        )
        author_label['label'] -= 1
        authors = pd.merge(authors, author_label, how='left', on='name')
        authors['label'] = authors['label'].fillna(-1).astype('int64')

        conf_label = self._read_file(
            join('label', 'googlescholar.8area.venue.label.txt'), ' ', ['name', 'label'], 'name'
        )
        conf_label['label'] -= 1
        confs = pd.merge(confs, conf_label, how='left', on='name')
        confs['label'] = confs['label'].fillna(-1).astype('int64')

        return authors, papers, confs, paper_author, paper_conf

    def _read_file(self, filename, sep, names, index_col=None, encoding='ISO-8859-1'):
        return pd.read_csv(
            join(self.raw_path, filename), sep=sep, names=names, index_col=index_col,
            encoding=encoding, quoting=csv.QUOTE_NONE
        )

    def has_cache(self):
        return os.path.exists(join(self.save_path, self.name + '_dgl_graph.bin')) \
               and os.path.exists(join(self.save_path, self.name + '_info.pkl'))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.g

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return 8


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
