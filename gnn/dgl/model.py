import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, HeteroGraphConv


class GCNFull(nn.Module):

    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GCN(nn.Module):

    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, blocks, inputs):
        h = self.conv1(blocks[0], inputs)
        h = F.relu(h)
        h = self.conv2(blocks[1], h)
        return h


class SAGEFull(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hid_feats, 'mean')
        self.conv2 = SAGEConv(hid_feats, out_feats, 'mean')

    def forward(self, g, inputs):
        # inputs are features of nodes
        h = F.relu(self.conv1(g, inputs))
        h = self.conv2(g, h)
        return h


class RGCNFull(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats) for rel in rel_names
        }, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats) for rel in rel_names
        }, aggregate='sum')

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h)
        return h


class RGCN(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats) for rel in rel_names
        }, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats) for rel in rel_names
        }, aggregate='sum')

    def forward(self, blocks, inputs):
        h = self.conv1(blocks[0], inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(blocks[1], h)
        return h


class DotProductPredictor(nn.Module):

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined in node_clf.py
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class MLPPredictor(nn.Module):

    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        score = self.W(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined in node_clf.py
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class HeteroDotProductPredictor(nn.Module):

    def forward(self, graph, h, etype):
        # h contains the node representations for each edge type computed from node_clf_hetero.py
        with graph.local_scope():
            graph.ndata['h'] = h  # assigns 'h' of all node types in one shot
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class HeteroMLPPredictor(nn.Module):

    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        score = self.W(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
        return {'score': score}

    def forward(self, graph, h, etype):
        # h contains the node representations for each edge type computed from node_clf_hetero.py
        with graph.local_scope():
            graph.ndata['h'] = h  # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']


class MarginLoss(nn.Module):

    def forward(self, pos_score, neg_score):
        return (1 - pos_score + neg_score.view(pos_score.shape[0], -1)).clamp(min=0).mean()
