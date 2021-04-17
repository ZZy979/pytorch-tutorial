import argparse

import dgl
import torch
from dgl.sampling import random_walk
from tqdm import trange

from gnn.data import AMinerCSDataset
from gnn.utils import metapath_adj


def main():
    parser = argparse.ArgumentParser(description='Metapath-based Random Walk for metapath2vec')
    parser.add_argument('--num-walks', type=int, default=1000, help='number of walks for each node')
    parser.add_argument('--walk-length', type=int, default=100, help='times to repeat metapath')
    parser.add_argument('output_file', help='output filename')
    args = parser.parse_args()

    data = AMinerCSDataset()
    g = data[0]

    ca = metapath_adj(g, ['cp', 'pa'])
    ca_c, ca_a = ca.nonzero()
    cag = dgl.heterograph(
        {('conf', 'ca', 'author'): (ca_c, ca_a), ('author', 'ac', 'conf'): (ca_a, ca_c)},
        {'conf': ca.shape[0], 'author': ca.shape[1]}
    )
    cag.edges['ca'].data['p'] = cag.edges['ac'].data['p'] = torch.from_numpy(ca.data).float()

    metapath = ['ca', 'ac']  # metapath = CAC, metapath*2 = CACAC
    f = open(args.output_file, 'w')
    for cid in trange(cag.num_nodes('conf'), ncols=80):
        traces, _ = random_walk(
            cag, [cid] * args.num_walks, metapath=metapath * args.walk_length, prob='p'
        )
        f.writelines([trace2name(data.author_names, data.conf_names, t) + '\n' for t in traces])
    f.close()


def trace2name(author_names, conf_names, trace):
    return ' '.join((author_names if i & 1 else conf_names)[trace[i]] for i in range(len(trace)))


if __name__ == '__main__':
    main()
