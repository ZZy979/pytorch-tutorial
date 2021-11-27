import dgl
import pandas as pd
import torch
from dgl.dataloading.negative_sampler import Uniform
from sklearn.preprocessing import LabelEncoder


class RatingKnowledgeGraphDataset:
    """基于知识图谱的用户评分数据集

    读取用户-物品评分数据和知识图谱三元组，并分别构造为两个图：

    * user_item_graph: DGLGraph (user, rate, item)二分图，由正向（评分大于等于阈值）的交互关系组成
    * knowledge_graph: DGLGraph 同构图，其中0~N_item-1对应user_item_graph中的item顶点（即物品集合是实体集合的子集）
      ，边特征relation表示关系类型
    """
    CONFIG = {
        'movie': {
            'rating_path': 'data/kgcn/movie/ratings.csv',
            'rating_sep': ',',
            'kg_path': 'data/kgcn/movie/kg.txt',
            'item2id_path': 'data/kgcn/movie/item_index2entity_id.txt',
            'threshold': 4.0
        },
        'music': {
            'rating_path': 'data/kgcn/music/user_artists.dat',
            'rating_sep': '\t',
            'kg_path': 'data/kgcn/music/kg.txt',
            'item2id_path': 'data/kgcn/music/item_index2entity_id.txt',
            'threshold': 0.0
        }
    }

    def __init__(self, dataset):
        self.dataset = dataset
        cfg = self.CONFIG[dataset]

        rating = pd.read_csv(
            cfg['rating_path'], sep=cfg['rating_sep'], names=['user_id', 'item_id', 'rating'],
            usecols=[0, 1, 2], skiprows=1
        )
        kg = pd.read_csv(cfg['kg_path'], sep='\t', names=['head', 'relation', 'tail'])
        item2entity = pd.read_csv(cfg['item2id_path'], sep='\t', names=['item_id', 'entity_id'])

        rating = rating[rating['item_id'].isin(item2entity['item_id'])]
        rating.reset_index(drop=True, inplace=True)
        rating['user_id'] = LabelEncoder().fit_transform(rating['user_id'])
        item2entity = dict(zip(item2entity['item_id'], item2entity['entity_id']))
        rating['item_id'] = rating['item_id'].apply(item2entity.__getitem__)
        rating['label'] = rating['rating'].apply(lambda r: int(r >= cfg['threshold']))
        rating = rating[rating['label'] == 1]
        user_item_graph = dgl.heterograph({
            ('user', 'rate', 'item'): (rating['user_id'].to_numpy(), rating['item_id'].to_numpy())
        })

        # 负采样
        neg_sampler = Uniform(1)
        nu, nv = neg_sampler(user_item_graph, torch.arange(user_item_graph.num_edges()))
        u, v = user_item_graph.edges()
        self.user_item_graph = dgl.heterograph({('user', 'rate', 'item'): (torch.cat([u, nu]), torch.cat([v, nv]))})
        self.user_item_graph.edata['label'] = torch.cat([torch.ones(u.shape[0]), torch.zeros(nu.shape[0])])

        kg['relation'] = LabelEncoder().fit_transform(kg['relation'])
        # 有重边，即两个实体之间可能存在多条边，关系类型不同
        knowledge_graph = dgl.graph((kg['head'], kg['tail']))
        knowledge_graph.edata['relation'] = torch.tensor(kg['relation'].tolist())
        self.knowledge_graph = dgl.add_reverse_edges(knowledge_graph, copy_edata=True)

    def get_num(self):
        return self.user_item_graph.num_nodes('user'), self.knowledge_graph.num_nodes(), self.knowledge_graph.edata['relation'].max().item() + 1
