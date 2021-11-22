"""异构图上使用邻居采样的顶点分类GNN

https://docs.dgl.ai/en/latest/guide/minibatch-node.html
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader

from gnn.data import UserItemDataset
from gnn.dgl.model import RGCN


def main():
    data = UserItemDataset()
    g = data[0]
    train_idx = g.nodes['user'].data['train_mask'].nonzero(as_tuple=True)[0]

    sampler = MultiLayerFullNeighborSampler(2)
    dataloader = NodeDataLoader(g, {'user': train_idx}, sampler, batch_size=256)

    model = RGCN(10, 20, 5, g.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(10):
        model.train()
        losses = []
        for input_nodes, output_nodes, blocks in dataloader:
            logits = model(blocks, blocks[0].srcdata['feat'])['user']
            loss = F.cross_entropy(logits, blocks[-1].dstnodes['user'].data['label'])
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch + 1, torch.tensor(losses).mean().item()))


if __name__ == '__main__':
    main()
