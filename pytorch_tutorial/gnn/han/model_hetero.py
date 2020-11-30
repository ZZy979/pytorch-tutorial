"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.
Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

from pytorch_tutorial.gnn.han.model import SemanticAttention


class HANLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        """HAN layer

        :param meta_paths: list of meta-paths, each as a list of edge types
        :param in_size: input feature dimension
        :param out_size: output feature dimension
        :param layer_num_heads: number of attention heads
        :param dropout: Dropout probability
        """
        super().__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(
                in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu,
                allow_zero_in_degree=True
            ))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = meta_paths

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        """
        :param g: The heterogeneous graph
        :param h: Input features
        :return: The output feature
        """
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                # 基于每条元路径的邻居构成的同构图
                self._cached_coalesced_graph[tuple(meta_path)] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[tuple(meta_path)]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HANHetero(nn.Module):

    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(
                meta_paths, hidden_size * num_heads[l - 1], hidden_size, num_heads[l], dropout
            ))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for han in self.layers:
            h = han(g, h)
        return self.predict(h)
