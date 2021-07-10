"""Label Propagation

* 论文链接：https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf
* DGL实现：https://github.com/dmlc/dgl/tree/master/examples/pytorch/label_propagation
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelPropagation(nn.Module):

    def __init__(self, num_layers, alpha):
        """标签传播模型

        .. math::
            Y^{(t+1)} = \\alpha D^{-1/2}AD^{-1/2}Y^{(t)} + (1-\\alpha)Y^{(t)}

        :param num_layers: int 传播层数
        :param alpha: float α参数
        """
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha

    @torch.no_grad()
    def forward(self, g, labels, mask=None):
        """
        :param g: DGLGraph 无向图
        :param labels: tensor(N) 标签
        :param mask: tensor(N), optional 有标签顶点mask
        :return: tensor(N, C) 预测标签概率
        """
        with g.local_scope():
            labels = F.one_hot(labels).float()  # (N, C)
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]
            else:
                y = labels

            degs = g.in_degrees().clamp(min=1)
            norm = torch.pow(degs, -0.5).unsqueeze(1)  # D^{-1/2}, (N, 1)

            for _ in range(self.num_layers):
                g.ndata['h'] = norm * y
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = self.alpha * norm * g.ndata.pop('h') + (1 - self.alpha) * y
            return y
