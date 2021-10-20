import os

import dgl
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from dgl.data import DGLDataset
from dgl.data.utils import makedirs, download, save_graphs, load_graphs, \
    generate_mask_tensor, idx2mask
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords

from gnn.utils import split_idx


class DBLPFourAreaDataset(DGLDataset):
    """4领域DBLP学术网络数据集，只有一个异构图

    统计数据
    -----
    * 顶点：4057 author, 14328 paper, 20 conf, 7723 term
    * 边：19645 paper-author, 14328 paper-conf, 85810 paper-term
    * 类别数：4
    * author顶点划分：800 train, 400 valid, 2857 test

    属性
    -----
    * num_classes: 类别数
    * metapaths: 使用的元路径
    * predict_ntype: 预测顶点类型

    author顶点属性
    -----
    * feat: tensor(4057, 334)，关键词的词袋表示（来自HAN作者预处理的数据集）
    * label: tensor(4057)，0: DB, 1: DM, 2: AI, 3: IR
    * train_mask, val_mask, test_mask: tensor(4057)

    conf顶点属性
    -----
    * label: tensor(20)，类别为0~3
    """
    _url = 'https://raw.githubusercontent.com/Jhy1993/HAN/master/data/DBLP_four_area/'
    _url2 = 'https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg'
    _raw_files = [
        'readme.txt', 'author_label.txt', 'paper.txt', 'conf_label.txt', 'term.txt',
        'paper_author.txt', 'paper_conf.txt', 'paper_term.txt'
    ]
    _seed = 42

    def __init__(self):
        super().__init__('DBLP_four_area', self._url)

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
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes['author'].data[k] = self.g.nodes['author'].data[k].bool()

    def process(self):
        self.authors, self.papers, self.confs, self.terms, \
            self.paper_author, self.paper_conf, self.paper_term = self._read_raw_data()
        self._filter_nodes_and_edges()
        self._lemmatize_terms()
        self._remove_stopwords()
        self._reset_index()

        self.g = self._build_graph()
        self._add_ndata()

    def _read_raw_data(self):
        authors = self._read_file('author_label.txt', names=['id', 'label', 'name'], index_col='id')
        papers = self._read_file('paper.txt', names=['id', 'title'], index_col='id', encoding='cp1252')
        confs = self._read_file('conf_label.txt', names=['id', 'label', 'name', 'dummy'], index_col='id')
        terms = self._read_file('term.txt', names=['id', 'name'], index_col='id')
        paper_author = self._read_file('paper_author.txt', names=['paper_id', 'author_id'])
        paper_conf = self._read_file('paper_conf.txt', names=['paper_id', 'conf_id'])
        paper_term = self._read_file('paper_term.txt', names=['paper_id', 'term_id'])
        return authors, papers, confs, terms, paper_author, paper_conf, paper_term

    def _read_file(self, filename, names, index_col=None, encoding='utf8'):
        return pd.read_csv(
            os.path.join(self.raw_path, filename), sep='\t', names=names, index_col=index_col,
            keep_default_na=False, encoding=encoding
        )

    def _filter_nodes_and_edges(self):
        """过滤掉不与学者关联的顶点和边"""
        self.paper_author = self.paper_author[self.paper_author['author_id'].isin(self.authors.index)]
        paper_ids = self.paper_author['paper_id'].drop_duplicates()
        self.papers = self.papers.loc[paper_ids]
        self.paper_conf = self.paper_conf[self.paper_conf['paper_id'].isin(paper_ids)]
        self.paper_term = self.paper_term[self.paper_term['paper_id'].isin(paper_ids)]
        self.terms = self.terms.loc[self.paper_term['term_id'].drop_duplicates()]

    def _lemmatize_terms(self):
        """对关键词进行词形还原并去重"""
        lemmatizer = WordNetLemmatizer()
        lemma_id_map, term_lemma_map = {}, {}
        for index, row in self.terms.iterrows():
            lemma = lemmatizer.lemmatize(row['name'])
            term_lemma_map[index] = lemma_id_map.setdefault(lemma, index)
        self.terms = pd.DataFrame(
            list(lemma_id_map.keys()), columns=['name'],
            index=pd.Index(lemma_id_map.values(), name='id')
        )
        self.paper_term.loc[:, 'term_id'] = [
            term_lemma_map[row['term_id']] for _, row in self.paper_term.iterrows()
        ]
        self.paper_term.drop_duplicates(inplace=True)

    def _remove_stopwords(self):
        """删除关键词中的停止词"""
        stop_words = sklearn_stopwords.union(nltk_stopwords.words('english'))
        self.terms = self.terms[~(self.terms['name'].isin(stop_words))]
        self.paper_term = self.paper_term[self.paper_term['term_id'].isin(self.terms.index)]

    def _reset_index(self):
        """将顶点id重置为0~n-1"""
        self.authors.reset_index(inplace=True)
        self.papers.reset_index(inplace=True)
        self.confs.reset_index(inplace=True)
        self.terms.reset_index(inplace=True)
        author_id_map = {row['id']: index for index, row in self.authors.iterrows()}
        paper_id_map = {row['id']: index for index, row in self.papers.iterrows()}
        conf_id_map = {row['id']: index for index, row in self.confs.iterrows()}
        term_id_map = {row['id']: index for index, row in self.terms.iterrows()}

        self.paper_author.loc[:, 'author_id'] = [author_id_map[i] for i in self.paper_author['author_id'].to_list()]
        self.paper_conf.loc[:, 'conf_id'] = [conf_id_map[i] for i in self.paper_conf['conf_id'].to_list()]
        self.paper_term.loc[:, 'term_id'] = [term_id_map[i] for i in self.paper_term['term_id'].to_list()]
        for df in (self.paper_author, self.paper_conf, self.paper_term):
            df.loc[:, 'paper_id'] = [paper_id_map[i] for i in df['paper_id']]

    def _build_graph(self):
        pa_p, pa_a = self.paper_author['paper_id'].to_list(), self.paper_author['author_id'].to_list()
        pc_p, pc_c = self.paper_conf['paper_id'].to_list(), self.paper_conf['conf_id'].to_list()
        pt_p, pt_t = self.paper_term['paper_id'].to_list(), self.paper_term['term_id'].to_list()
        return dgl.heterograph({
            ('paper', 'pa', 'author'): (pa_p, pa_a),
            ('author', 'ap', 'paper'): (pa_a, pa_p),
            ('paper', 'pc', 'conf'): (pc_p, pc_c),
            ('conf', 'cp', 'paper'): (pc_c, pc_p),
            ('paper', 'pt', 'term'): (pt_p, pt_t),
            ('term', 'tp', 'paper'): (pt_t, pt_p)
        })

    def _add_ndata(self):
        _raw_file2 = os.path.join(self.raw_dir, 'DBLP4057_GAT_with_idx.mat')
        if not os.path.exists(_raw_file2):
            raise FileNotFoundError('请手动下载文件 {} 提取码：6b3h 并保存到 {}'.format(
                self._url2, _raw_file2
            ))
        mat = sio.loadmat(_raw_file2)
        self.g.nodes['author'].data['feat'] = torch.from_numpy(mat['features']).float()
        self.g.nodes['author'].data['label'] = torch.tensor(self.authors['label'].to_list())

        n_authors = len(self.authors)
        train_idx, val_idx, test_idx = split_idx(np.arange(n_authors), 800, 400, self._seed)
        self.g.nodes['author'].data['train_mask'] = generate_mask_tensor(idx2mask(train_idx, n_authors))
        self.g.nodes['author'].data['val_mask'] = generate_mask_tensor(idx2mask(val_idx, n_authors))
        self.g.nodes['author'].data['test_mask'] = generate_mask_tensor(idx2mask(test_idx, n_authors))

        self.g.nodes['conf'].data['label'] = torch.tensor(self.confs['label'].to_list())

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

    @property
    def metapaths(self):
        return [['ap', 'pa'], ['ap', 'pc', 'cp', 'pa'], ['ap', 'pt', 'tp', 'pa']]

    @property
    def predict_ntype(self):
        return 'author'


class DBLP4057Dataset(DGLDataset):
    """HAN作者处理的DBLP数据集：https://github.com/Jhy1993/HAN#datasets

    只有一个样本，包括author顶点基于APA, APCPA和APTPA三个元路径的邻居组成的同构图

    >>> data = DBLP4057Dataset()
    >>> apa_g, apcpa_g, aptpa_g = data[0]

    统计数据
    -----
    * apa_g: 4057个顶点，11113条边
    * apcpa_g: 4057个顶点，5000495条边
    * aptpa_g: 4057个顶点，6772278条边
    * 类别数：4
    * 划分：800 train, 400 valid, 2857 test

    顶点属性
    -----
    * feat: tensor(4057, 334)
    * label: tensor(4057)
    * train_mask, val_mask, test_mask: tensor(4057)
    """

    def __init__(self):
        super().__init__('DBLP4057', 'https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg')

    def download(self):
        file_path = os.path.join(self.raw_dir, 'DBLP4057_GAT_with_idx.mat')
        if not os.path.exists(file_path):
            raise FileNotFoundError('请手动下载文件 {} 提取码：6b3h 并保存为 {}'.format(
                self.url, file_path
            ))

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), self.gs)

    def load(self):
        self.gs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        for g in self.gs:
            for k in ('train_mask', 'val_mask', 'test_mask'):
                g.ndata[k] = g.ndata[k].bool()

    def process(self):
        data = sio.loadmat(os.path.join(self.raw_dir, 'DBLP4057_GAT_with_idx.mat'))
        apa_g = dgl.graph(data['net_APA'].nonzero())
        apcpa_g = dgl.graph(data['net_APCPA'].nonzero())
        aptpa_g = dgl.graph(data['net_APTPA'].nonzero())
        self.gs = [apa_g, apcpa_g, aptpa_g]

        features = torch.from_numpy(data['features']).float()
        labels = torch.from_numpy(data['label'].nonzero()[1])
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
        return 4
