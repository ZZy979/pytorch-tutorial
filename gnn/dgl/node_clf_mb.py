"""使用邻居采样的顶点分类GNN

https://docs.dgl.ai/en/latest/guide/minibatch-node.html
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import CiteseerGraphDataset
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader

from gnn.dgl.model import GCN


def main():
    data = CiteseerGraphDataset()
    g = data[0]
    train_idx = g.ndata['train_mask'].nonzero(as_tuple=True)[0]

    sampler = MultiLayerFullNeighborSampler(2)
    dataloader = NodeDataLoader(g, train_idx, sampler, batch_size=32)

    model = GCN(g.ndata['feat'].shape[1], 100, data.num_classes)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(30):
        model.train()
        losses = []
        for input_nodes, output_nodes, blocks in dataloader:
            logits = model(blocks, blocks[0].srcdata['feat'])
            loss = F.cross_entropy(logits, blocks[-1].dstdata['label'])
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch + 1, torch.tensor(losses).mean().item()))


if __name__ == '__main__':
    main()
