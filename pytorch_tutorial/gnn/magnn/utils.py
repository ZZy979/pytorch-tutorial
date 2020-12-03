from collections import defaultdict

import dgl
import torch


def metapath_based_graph(g, metapath):
    """返回异构图基于给定元路径的邻居构成的图和元路径实例。

    与dgl.metapath_reachable_graph()的区别：如果两个顶点之间有多个元路径实例则在返回的图中有多条边

    :param g: DGLHeteroGraph 异构图
    :param metapath: List[(str, str, str)] 元路径，边类型列表
    :return: DGLGraph, List[List[int]] 基于元路径的图，所有元路径实例
    """
    instances = metapath_instances(g, metapath)
    src_nodes, dst_nodes = [], []
    for path in instances:
        src_nodes.append(path[0])
        dst_nodes.append(path[-1])
    mg = dgl.graph((src_nodes, dst_nodes))
    return mg, instances


def metapath_instance_feat(g, metapath, instances, feat):
    """返回异构图基于给定元路径的邻居构成的图的边（元路径实例）特征，元路径实例特征由中间顶点的特征组成

    :param g: DGLHeteroGraph 原始异构图
    :param metapath: List[(str, str, str)] 元路径，边类型列表
    :param instances: List[List[int]] 元路径实例列表，每个实例为中间顶点id列表
    :param feat: str 顶点属性名称，所有类型的顶点的属性应具有相同的维度d
    :return: tensor(E, L, d) 元路径实例特征，L为元路径长度
    """
    ntypes = [metapath[0][0]] + [etype[2] for etype in metapath]
    edata = [
        [g.nodes[ntype].data[feat][v].tolist() for v, ntype in zip(path, ntypes)]
        for path in instances
    ]
    return torch.tensor(edata)


def metapath_instances(g, metapath):
    """返回异构图中给定元路径的所有实例。

    :param g: DGLHeteroGraph 异构图
    :param metapath: List[(str, str, str)] 元路径，边类型列表
    :return: tensor(N, L) N是元路径实例个数，L是元路径长度
    """
    if any(metapath[i][0] != metapath[i - 1][2] for i in range(1, len(metapath))):
        return []
    paths = g.nodes(metapath[0][0]).unsqueeze(1).tolist()
    for etype in metapath:
        new_paths = []
        neighbors = etype_neighbors(g, etype)
        for path in paths:
            for neighbor in neighbors[path[-1]]:
                new_paths.append(path + [neighbor])
        paths = new_paths
    return paths


def etype_neighbors(g, etype):
    """返回异构图中给定基于边类型的邻居。

    :param g: DGLHeteroGraph 异构图
    :param etype: (str, str, str) 边类型
    :return: Dict[int, List[int]] 每个源顶点基于该类型的边的邻居
    """
    stype = g.to_canonical_etype(etype)[0]
    src_nodes, dst_nodes = g.out_edges(g.nodes(stype), etype=etype)
    neighbors = defaultdict(list)
    for u, v in zip(src_nodes.tolist(), dst_nodes.tolist()):
        neighbors[u].append(v)
    return neighbors
