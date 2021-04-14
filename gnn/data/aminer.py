import csv
import os
import zipfile
from os.path import join

import dgl
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import download, save_graphs, save_info, load_graphs, load_info


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
