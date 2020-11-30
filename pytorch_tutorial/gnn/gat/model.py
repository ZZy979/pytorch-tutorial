"""Graph Attention Networks (GAT)

* 论文链接：<https://arxiv.org/abs/1710.10903>
* 官方代码：<https://github.com/PetarV-/GAT>
* DGL实现：<https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat>
"""
import torch.nn as nn
from dgl.nn import GATConv


class GAT(nn.Module):

    def __init__(
            self, g, num_layers, in_dim, hidden_dim, n_classes,
            heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super().__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, hidden_dim, heads[0], feat_drop, attn_drop,
            negative_slope, False, self.activation
        ))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                hidden_dim * heads[l - 1], hidden_dim, heads[l], feat_drop, attn_drop,
                negative_slope, residual, self.activation
            ))
        # output projection
        self.gat_layers.append(GATConv(
            hidden_dim * heads[-2], n_classes, heads[-1], feat_drop, attn_drop,
            negative_slope, residual, None
        ))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        return self.gat_layers[-1](self.g, h).mean(1)
