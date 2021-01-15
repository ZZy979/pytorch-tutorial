"""异构图上使用邻居采样的顶点分类GNN

https://docs.dgl.ai/guide/minibatch-node.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeCollator, DataLoader
from dgl.nn import HeteroGraphConv, GraphConv

from gnn.dgl.node_clf_hetero import build_user_item_graph


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


def main():
    g = build_user_item_graph()
    train_idx = g.nodes['user'].data['train_mask'].nonzero(as_tuple=False)

    sampler = MultiLayerFullNeighborSampler(2)
    collator = NodeCollator(g, {'user': train_idx}, sampler)
    dataloader = DataLoader(collator.dataset, 32, collate_fn=collator.collate)

    model = RGCN(10, 20, 5, g.etypes)
    opt = optim.Adam(model.parameters())

    for epoch in range(5):
        model.train()
        losses = []
        for input_nodes, output_nodes, blocks in dataloader:
            logits = model(blocks, blocks[0].srcdata['feature'])['user']
            loss = F.cross_entropy(logits, blocks[-1].dstnodes['user'].data['label'])
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch + 1, torch.tensor(losses).mean().item()))


if __name__ == '__main__':
    main()
