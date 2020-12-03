import dgl
import torch
from dgl.data import DGLDataset


class ToyAcademicDataset(DGLDataset):

    def __init__(self):
        super().__init__('toy-academic')

    def process(self):
        pa_p, pa_a = [0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 2, 3]
        pc_p, pc_c = [0, 1, 2], [0, 1, 1]
        self.g = dgl.heterograph({
            ('paper', 'pa', 'author'): (pa_p, pa_a),
            ('author', 'ap', 'paper'): (pa_a, pa_p),
            ('paper', 'pc', 'conf'): (pc_p, pc_c),
            ('conf', 'cp', 'paper'): (pc_c, pc_p)
        })
        for ntype in self.g.ntypes:
            self.g.nodes[ntype].data['feat'] = torch.eye(self.g.num_nodes(ntype))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.g

    def __len__(self):
        return 1
