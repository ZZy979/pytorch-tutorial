from collections import defaultdict

import dgl
import torch


def metapath_based_graph(g, metapath, nids=None, edge_feat_name='inst'):
    """返回异构图基于给定元路径的邻居组成的图，元路径实例作为边特征，如果元路径是对称的则返回的是同构图，否则是二分图。

    与dgl.metapath_reachable_graph()的区别：如果两个顶点之间有多个元路径实例则在返回的图中有多条边

    :param g: DGLGraph 异构图
    :param metapath: List[str or (str, str, str)] 元路径，边类型列表
    :param nids: tensor(N), optional 起点id，如果为None则选择该类型所有顶点
    :param edge_feat_name: str 保存元路径实例的边特征名称
    :return: DGLGraph 基于元路径的图
    """
    instances = metapath_instances(g, metapath, nids)
    src_nodes, dst_nodes = instances[:, 0], instances[:, -1]
    src_type, dst_type = g.to_canonical_etype(metapath[0])[0], g.to_canonical_etype(metapath[-1])[2]
    if src_type == dst_type:
        mg = dgl.graph((src_nodes, dst_nodes), num_nodes=g.num_nodes(src_type))
        mg.edata[edge_feat_name] = instances
    else:
        mg = dgl.heterograph(
            {(src_type, '_E', dst_type): (src_nodes, dst_nodes)},
            {src_type: g.num_nodes(src_type), dst_type: g.num_nodes(dst_type)}
        )
        mg.edges['_E'].data[edge_feat_name] = instances
    return mg


def metapath_instances(g, metapath, nids=None):
    """返回异构图中给定元路径的所有实例。

    :param g: DGLGraph 异构图
    :param metapath: List[str or (str, str, str)] 元路径，边类型列表
    :param nids: tensor(N), optional 起点id，如果为None则选择该类型所有顶点
    :return: tensor(E, L) E为元路径实例个数，L为元路径长度
    """
    src_type = g.to_canonical_etype(metapath[0])[0]
    if nids is None:
        nids = g.nodes(src_type)
    paths = nids.unsqueeze(1).tolist()
    for etype in metapath:
        new_paths = []
        neighbors = etype_neighbors(g, etype)
        for path in paths:
            for neighbor in neighbors[path[-1]]:
                new_paths.append(path + [neighbor])
        paths = new_paths
    return torch.tensor(paths, dtype=torch.long)


def etype_neighbors(g, etype):
    """返回异构图中给定基于边类型的邻居。

    :param g: DGLGraph 异构图
    :param etype: (str, str, str) 规范边类型
    :return: Dict[int, List[int]] 每个源顶点基于该类型的边的邻居
    """
    adj = g.adj(scipy_fmt='coo', etype=etype)
    neighbors = defaultdict(list)
    for u, v in zip(adj.row, adj.col):
        neighbors[u].append(v)
    return neighbors


def to_ntype_list(g, metapath):
    """将边类型列表表示的元路径转换为顶点类型列表。

    例如：['ap', 'pc', 'cp', 'pa] -> ['a', 'p', 'c', 'p', 'a']

    :param g: DGLGraph 异构图
    :param metapath: List[str or (str, str, str)] 元路径，边类型列表
    :return: List[str] 元路径的顶点类型列表表示
    """
    metapath = [g.to_canonical_etype(etype) for etype in metapath]
    return [metapath[0][0]] + [etype[2] for etype in metapath]


def metapath_instance_feat(metapath, node_feats, instances):
    """返回元路径实例特征，由中间顶点的特征组成。

    :param metapath: List[str] 元路径，顶点类型列表
    :param node_feats: Dict[str, tendor(N_i, d)] 顶点类型到顶点特征的映射，所有类型顶点的特征应具有相同的维数d
    :param instances: tensor(E, L) 元路径实例，E为元路径实例个数，L为元路径长度
    :return: tensor(E, L, d) 元路径实例特征
    """
    feat_dim = node_feats[metapath[0]].shape[1]
    inst_feat = torch.zeros(instances.shape + (feat_dim,))
    for i, ntype in enumerate(metapath):
        inst_feat[:, i] = node_feats[ntype][instances[:, i]]
    return inst_feat
