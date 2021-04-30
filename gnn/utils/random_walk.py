import torch
from dgl.sampling import random_walk
from torch.utils.data import DataLoader
from tqdm import tqdm


def metapath_random_walk(g, metapaths, num_walks, walk_length, output_file):
    """基于元路径的随机游走

    :param g: DGLGraph 异构图
    :param metapaths: Dict[str, List[str]] 顶点类型到元路径的映射
    :param num_walks: int 每个顶点游走次数
    :param walk_length: int 元路径重复次数
    :param output_file: str 输出文件名
    :return:
    """
    f = open(output_file, 'w')
    for ntype, metapath in metapaths.items():
        print(ntype)
        loader = DataLoader(torch.arange(g.num_nodes(ntype)), batch_size=200)
        for b in tqdm(loader, ncols=80):
            nodes = torch.repeat_interleave(b, num_walks)
            traces, types = random_walk(g, nodes, metapath=metapath * walk_length)
            f.writelines([trace2name(g, trace, types) + '\n' for trace in traces])
    f.close()


def trace2name(g, trace, types):
    return ' '.join(g.ntypes[t] + '_' + str(int(n)) for n, t in zip(trace, types) if int(n) >= 0)
