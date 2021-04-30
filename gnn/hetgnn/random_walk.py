import argparse

from gnn.data import AMiner2Dataset
from gnn.utils import metapath_random_walk


def main():
    parser = argparse.ArgumentParser(description='Random walk with restart')
    parser.add_argument('--num-walks', type=int, default=10, help='number of walks for each node')
    parser.add_argument('--walk-length', type=int, default=30, help='times to repeat metapath')
    parser.add_argument('output_file', help='output filename')
    args = parser.parse_args()

    data = AMiner2Dataset()
    g = data[0]

    # author_0特殊处理，添加假边
    g.add_edges([0] * 10, list(range(10)), etype='write')
    g.add_edges(list(range(10)), [0] * 10, etype='write_rev')

    metapaths = {
        'author': ['write', 'publish', 'publish_rev', 'write_rev'],  # APVPA
        'paper': ['write_rev', 'write', 'publish', 'publish_rev'],  # PAPVP
        'venue': ['publish_rev', 'write_rev', 'write', 'publish']  # VPAPV
    }
    metapath_random_walk(g, metapaths, args.num_walks, args.walk_length, args.output_file)


if __name__ == '__main__':
    main()
