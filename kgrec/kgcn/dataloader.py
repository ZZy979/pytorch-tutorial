import dgl
from dgl.dataloading import EdgeCollator, EdgeDataLoader
from dgl.utils import prepare_tensor
from torch.utils.data import DataLoader


class KGCNEdgeCollator(EdgeCollator):

    def __init__(self, user_item_graph, eids, block_sampler, knowledge_graph, negative_sampler=None):
        """用于KGCN的EdgeCollator

        :param user_item_graph: DGLGraph 用户-物品图
        :param eids: tensor(E) 训练边id
        :param block_sampler: BlockSampler 邻居采样器
        :param knowledge_graph: DGLGraph 知识图谱
        :param negative_sampler: 负采样器
        """
        super().__init__(user_item_graph, eids, block_sampler, knowledge_graph, negative_sampler=negative_sampler)

    def _collate(self, items):
        """根据边id采样子图

        :param items: tensor(B) 边id
        :return: tensor(N_src), DGLGraph, List[DGLBlock] 知识图谱的输入顶点id，用户-物品图的边子图，
        知识图谱根据边id关联的物品id采样的多层MFG
        """
        items = prepare_tensor(self.g_sampling, items, 'items')
        pair_graph = dgl.edge_subgraph(self.g, items)
        seed_nodes = pair_graph.ndata[dgl.NID]
        blocks = self.block_sampler.sample_blocks(self.g_sampling, seed_nodes['item'])
        input_nodes = blocks[0].srcdata[dgl.NID]
        return input_nodes, pair_graph, blocks

    def _collate_with_negative_sampling(self, items):
        """根据边id采样子图，并进行负采样

        :param items: tensor(B) 边id
        :return: tensor(N_src), DGLGraph, DGLGraph, List[DGLBlock] 知识图谱的输入顶点id，用户-物品图的边子图，负样本图，
        知识图谱根据边id关联的物品id采样的多层MFG
        """
        items = prepare_tensor(self.g_sampling, items, 'items')
        pair_graph = dgl.edge_subgraph(self.g, items, relabel_nodes=False)
        induced_edges = pair_graph.edata[dgl.EID]

        neg_srcdst = self.negative_sampler(self.g, items)
        neg_pair_graph = dgl.heterograph({self.g.canonical_etypes[0]: neg_srcdst})

        pair_graph, neg_pair_graph = dgl.compact_graphs([pair_graph, neg_pair_graph])
        pair_graph.edata[dgl.EID] = induced_edges
        seed_nodes = pair_graph.ndata[dgl.NID]

        blocks = self.block_sampler.sample_blocks(self.g_sampling, seed_nodes['item'])
        input_nodes = blocks[0].srcdata[dgl.NID]
        return input_nodes, pair_graph, neg_pair_graph, blocks


class KGCNEdgeDataLoader(EdgeDataLoader):

    def __init__(self, user_item_graph, eids, block_sampler, knowledge_graph, negative_sampler=None, device='cpu', **kwargs):
        """用于KGCN的EdgeDataLoader

        :param user_item_graph: DGLGraph 用户-物品图
        :param eids: tensor(E) 训练边id
        :param block_sampler: BlockSampler 邻居采样器
        :param knowledge_graph: DGLGraph 知识图谱
        :param device: torch.device
        :param kwargs: DataLoader的其他参数
        """
        super().__init__(user_item_graph, eids, block_sampler, device, **kwargs)
        self.collator = KGCNEdgeCollator(user_item_graph, eids, block_sampler, knowledge_graph, negative_sampler)
        self.dataloader = DataLoader(self.collator.dataset, collate_fn=self.collator.collate, **kwargs)
