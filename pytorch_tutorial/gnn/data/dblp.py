import os
from collections import Counter

import dgl
import scipy.io as sio
import torch
from dgl.data import DGLDataset
from dgl.data.utils import makedirs, download, generate_mask_tensor, idx2mask
from gensim.parsing.preprocessing import STOPWORDS


def split_idx(labels, train_size, val_size):
    count = Counter(labels.tolist())
    train_idx, val_idx, test_idx = [], [], []
    for c, n in count.items():
        n_train, n_val = int(n * train_size), int(n * val_size)
        idx = torch.nonzero(labels == c, as_tuple=False).squeeze(1).tolist()
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return train_idx, val_idx, test_idx


class DBLPFourAreaDataset(DGLDataset):
    """4个领域的DBLP学术网络数据集，只有一个图

    统计数据
    -----
    * 顶点：4057 author, 14376 paper, 20 conf, 8718 term
    * 边：19645 paper-author, 14376 paper-conf, 85825 paper-term

    属性
    -----
    * num_classes: 类别数(4)
    * author_name, paper_name, conf_name, term_name: List[str]，分别为每种顶点的名字

    author顶点属性
    -----
    * feat: tensor(4057, 331)，关键词的词袋表示
    * label: tensor(4057)，类别为0~3 (0: DB, 1: DM, 2: AI, 3: IR)
    * train_mask, val_mask, test_mask: tensor(4057)，True的数量分别为810, 403, 2844

    paper顶点属性
    -----
    * feat: tensor(14376, 331)，关键词的词袋表示
    * label: tensor(14376)，类别为0~3
    * train_mask, val_mask, test_mask: tensor(14376)，True的数量分别为2873, 1436, 10067

    conf顶点属性
    -----
    * label: tensor(20)，类别为0~3
    """
    files = [
        'readme.txt', 'author_label.txt', 'paper.txt', 'conf_label.txt', 'term.txt',
        'paper_author.txt', 'paper_conf.txt', 'paper_term.txt'
    ]

    def __init__(self):
        super().__init__(
            'DBLP_four_area', 'https://github.com/Jhy1993/HAN/raw/master/data/DBLP_four_area/'
        )

    def download(self):
        if not os.path.exists(self.raw_path):
            makedirs(self.raw_path)
        for file in self.files:
            download(self.url + file, os.path.join(self.raw_path, file))

    def save(self):
        dgl.save_graphs(self._cache_file, [self.g])

    def load(self):
        graphs, _ = dgl.load_graphs(self._cache_file)
        self.g = graphs[0]
        for ntype in ('author', 'paper'):
            for k in ('train_mask', 'val_mask', 'test_mask'):
                self.g.nodes['author'].data[k] = self.g.nodes['author'].data[k].type(torch.bool)

    def process(self):
        author_labels = self.read_author()
        conf_labels = self.read_conf()
        self.read_papers()
        self.read_terms()

        pa_p, pa_a = self.read_edge('paper_author.txt', self.paper_id, self.author_id)
        pc_p, pc_c = self.read_edge('paper_conf.txt', self.paper_id, self.conf_id)
        pt_p, pt_t = self.read_edge('paper_term.txt', self.paper_id, self.term_id)
        self.g = dgl.heterograph({
            ('paper', 'pa', 'author'): (pa_p, pa_a),
            ('author', 'ap', 'paper'): (pa_a, pa_p),
            ('paper', 'pc', 'conf'): (pc_p, pc_c),
            ('conf', 'cp', 'paper'): (pc_c, pc_p),
            ('paper', 'pt', 'term'): (pt_p, pt_t),
            ('term', 'tp', 'paper'): (pt_t, pt_p)
        })

        selected_terms = torch.nonzero(self.g.in_degrees(etype='pt') >= 50, as_tuple=False).squeeze(1)
        author_term_adj = self.g.adj(etype='ap').to_dense() @ self.g.adj(etype='pt').to_dense()
        author_train_idx, author_val_idx, author_test_idx = split_idx(author_labels, 0.2, 0.1)
        n_authors = len(self.author_id)
        self.g.nodes['author'].data['feat'] = author_term_adj[:, selected_terms]
        self.g.nodes['author'].data['label'] = author_labels
        self.g.nodes['author'].data['train_mask'] = generate_mask_tensor(idx2mask(author_train_idx, n_authors))
        self.g.nodes['author'].data['val_mask'] = generate_mask_tensor(idx2mask(author_val_idx, n_authors))
        self.g.nodes['author'].data['test_mask'] = generate_mask_tensor(idx2mask(author_test_idx, n_authors))

        paper_labels = conf_labels[pc_c]
        paper_train_idx, paper_val_idx, paper_test_idx = split_idx(paper_labels, 0.2, 0.1)
        n_papers = len(self.paper_id)
        self.g.nodes['paper'].data['feat'] = self.g.adj(etype='pt').to_dense()[:, selected_terms]
        self.g.nodes['paper'].data['label'] = paper_labels
        self.g.nodes['paper'].data['train_mask'] = generate_mask_tensor(idx2mask(paper_train_idx, n_papers))
        self.g.nodes['paper'].data['val_mask'] = generate_mask_tensor(idx2mask(paper_val_idx, n_papers))
        self.g.nodes['paper'].data['test_mask'] = generate_mask_tensor(idx2mask(paper_test_idx, n_papers))

        self.g.nodes['conf'].data['label'] = conf_labels

    def read_author(self):
        self.author_id, self.author_name = {}, []
        aid = 0
        labels = []
        with open(os.path.join(self.raw_path, 'author_label.txt')) as f:
            for line in f:
                line = line.strip().split('\t')
                self.author_id[int(line[0])] = aid
                labels.append(int(line[1]))
                self.author_name.append(line[2])
                aid += 1
        return torch.tensor(labels)

    def read_conf(self):
        self.conf_id, self.conf_name = {}, []
        cid = 0
        labels = []
        with open(os.path.join(self.raw_path, 'conf_label.txt')) as f:
            for line in f:
                line = line.strip().split('\t')
                self.conf_id[int(line[0])] = cid
                labels.append(int(line[1]))
                self.conf_name.append(line[2])
                cid += 1
        return torch.tensor(labels)

    def read_papers(self):
        self.paper_id, self.paper_name = {}, []
        pid = 0
        with open(os.path.join(self.raw_path, 'paper.txt')) as f:
            for line in f:
                line = line.strip().split('\t')
                self.paper_id[int(line[0])] = pid
                self.paper_name.append(line[1])
                pid += 1

    def read_terms(self):
        self.term_id, self.term_name = {}, []
        tid = 0
        with open(os.path.join(self.raw_path, 'term.txt')) as f:
            for line in f:
                line = line.strip().split('\t')
                if line[1] not in STOPWORDS:
                    self.term_id[int(line[0])] = tid
                    self.term_name.append(line[1])
                    tid += 1

    def read_edge(self, file, src_id, dst_id):
        src, dst = [], []
        with open(os.path.join(self.raw_path, file)) as f:
            for line in f:
                line = line.strip().split('\t')
                u, v = int(line[0]), int(line[1])
                if u in src_id and v in dst_id:
                    src.append(src_id[u])
                    dst.append(dst_id[v])
        return src, dst

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
        return 4

    @property
    def meta_paths(self):
        return [['ap', 'pa'], ['ap', 'pc', 'cp', 'pa'], ['ap', 'pt', 'tp', 'pa']]

    @property
    def _cache_file(self):
        return os.path.join(self.save_path, self.name + '.bin')


class DBLP4057Dataset(DGLDataset):
    """HAN作者处理的ACM数据集：https://github.com/Jhy1993/HAN#datasets

    只有一个图，由author顶点基于APA, APCPA和APTPA三个元路径的邻居的同构图组成

    >>> data = DBLP4057Dataset()
    >>> apa_g, apcpa_g, aptpa_g = data[0]

    * apa_g: 4057个顶点，11113条边
    * apcpa_g: 4057个顶点，5000495条边
    * aptpa_g: 4057个顶点，6772278条边

    三个图都有以下顶点属性：

    * feat: tensor(4057, 334)
    * label: tensor(4057)，类别为0~3
    * train_mask, val_mask, test_mask: tensor(3025)，True的数量分别为800, 400, 2857
    """

    def __init__(self):
        super().__init__('DBLP4057', 'https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg')

    def download(self):
        if not os.path.exists(self._raw_file):
            raise FileNotFoundError('请手动下载文件 {} 提取码：6b3h 并保存为 {}'.format(
                self.url, os.path.join(self._raw_file)
            ))

    def save(self):
        dgl.save_graphs(self._cache_file, self.gs)

    def load(self):
        self.gs, _ = dgl.load_graphs(self._cache_file)
        for g in self.gs:
            for k in ('train_mask', 'val_mask', 'test_mask'):
                g.ndata[k] = g.ndata[k].type(torch.bool)

    def process(self):
        data = sio.loadmat(self._raw_file)
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
        return os.path.exists(self._cache_file)

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.gs

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return 4

    @property
    def _raw_file(self):
        return os.path.join(self.raw_dir, self.name + '.mat')

    @property
    def _cache_file(self):
        return os.path.join(self.save_dir, self.name + '.bin')
