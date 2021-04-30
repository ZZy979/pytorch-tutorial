from collections import Counter
from itertools import chain

import dgl
import dgl.function as fn
import torch
from gensim.models import Word2Vec

from gnn.data import AMiner2Dataset


def load_data(neighbor_path, node_embed_path):
    data = AMiner2Dataset()
    g = data[0]
    # author_0特殊处理，添加假边
    g.add_edges(0, 0, etype='write')
    g.add_edges(0, 0, etype='write_rev')

    print('Loading pretrained node embeddings...')
    load_pretrained_node_embed(g, node_embed_path)
    print('Propagating input features...')
    feats = propagate_feature(g)
    print('Constructing neighbor graph...')
    ng = construct_neighbor_graph(g, neighbor_path, {'author': 10, 'paper': 10, 'venue': 3})
    return ng, feats


def load_pretrained_node_embed(g, node_embed_path):
    model = Word2Vec.load(node_embed_path)
    for ntype in g.ntypes:
        g.nodes[ntype].data['net_embed'] = torch.from_numpy(
            model.wv[[f'{ntype}_{i}' for i in range(g.num_nodes(ntype))]]
        )


def propagate_feature(g):
    with g.local_scope():
        g.multi_update_all({
            'write': (fn.copy_u('net_embed', 'm'), fn.mean('m', 'a_net_embed')),
            'publish_rev': (fn.copy_u('net_embed', 'm'), fn.mean('m', 'v_net_embed'))
        }, 'sum')
        paper_feats = torch.stack([
            g.nodes['paper'].data[k] for k in
            ('abstract_embed', 'title_embed', 'net_embed', 'v_net_embed', 'a_net_embed')
        ], dim=1)  # (N_p, 5, d)

        ap = find_neighbors(g, 'write_rev', 3).view(1, -1)  # (1, 3N_a)
        ap_abstract_embed = g.nodes['paper'].data['abstract_embed'][ap] \
            .view(g.num_nodes('author'), 3, -1)  # (N_a, 3, d)
        author_feats = torch.cat([
            g.nodes['author'].data['net_embed'].unsqueeze(dim=1), ap_abstract_embed
        ], dim=1)  # (N_a, 4, d)

        vp = find_neighbors(g, 'publish', 5).view(1, -1)  # (1, 5N_v)
        vp_abstract_embed = g.nodes['paper'].data['abstract_embed'][vp] \
            .view(g.num_nodes('venue'), 5, -1)  # (N_v, 5, d)
        venue_feats = torch.cat([
            g.nodes['venue'].data['net_embed'].unsqueeze(dim=1), vp_abstract_embed
        ], dim=1)  # (N_v, 6, d)

        return {'author': author_feats, 'paper': paper_feats, 'venue': venue_feats}


def find_neighbors(g, etype, n):
    num_nodes = g.num_nodes(g.to_canonical_etype(etype)[2])
    u, v = g.in_edges(torch.arange(num_nodes), etype=etype)
    neighbors = [[] for _ in range(num_nodes)]
    for i in range(len(v)):
        neighbors[v[i].item()].append(u[i].item())
    for v in range(num_nodes):
        if len(neighbors[v]) < n:
            neighbors[v] += [neighbors[v][-1]] * (n - len(neighbors[v]))
        elif len(neighbors[v]) > n:
            neighbors[v] = neighbors[v][:n]
    return torch.tensor(neighbors)  # (N_dst, n)


def construct_neighbor_graph(g, neighbor_path, neighbor_size):
    counts = {
        f'{stype}-{dtype}': [Counter() for _ in range(g.num_nodes(dtype))]
        for stype in g.ntypes for dtype in g.ntypes
    }
    with open(neighbor_path) as f:
        for line in f:
            center, neighbors = line.strip().split(' ', 1)
            dtype, v = parse_node_name(center)
            neighbors = neighbors.split(' ')
            for n in neighbors:
                stype, u = parse_node_name(n)
                counts[f'{stype}-{dtype}'][v][u] += 1

    edges = {}
    for dtype in g.ntypes:
        for stype in g.ntypes:
            etype = f'{stype}-{dtype}'
            edges[(stype, etype, dtype)] = (torch.tensor(list(chain.from_iterable(
                (u for u, _ in counts[etype][v].most_common(neighbor_size[stype]))
                for v in range(g.num_nodes(dtype))
            ))), torch.arange(g.num_nodes(dtype)).repeat_interleave(neighbor_size[stype]))

    ng = dgl.heterograph(edges, {ntype: g.num_nodes(ntype) for ntype in g.ntypes})
    for dtype in ng.ntypes:
        for stype in ng.ntypes:
            assert torch.all(ng.in_degrees(etype=f'{stype}-{dtype}') == neighbor_size[stype]), \
                f'not all in-degrees of edge type {stype}-{dtype} are {neighbor_size[stype]}'
    return ng


def parse_node_name(node):
    ntype, nid = node.split('_')
    return ntype, int(nid)


def construct_neg_graph(g, neg_sampler):
    return dgl.heterograph(
        neg_sampler(g, {etype: torch.arange(g.num_edges(etype)) for etype in g.etypes}),
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    )
