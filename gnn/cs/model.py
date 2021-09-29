"""Correct and Smooth (C&S)

论文链接：https://arxiv.org/pdf/2010.13993
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.0):
        super().__init__()
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.linears.append(nn.Linear(in_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = F.relu(x)
            x = self.batch_norms[i](x)
            x = self.dropout(x)
        x = self.linears[-1](x)
        return x


class LabelPropagation(nn.Module):

    def __init__(self, num_layers, alpha, norm):
        """标签传播模型

        .. math::
            Y^{(t+1)} = \\alpha SY^{(t)} + (1-\\alpha)Y, Y^{(0)} = Y

        :param num_layers: int 传播层数
        :param alpha: float α参数
        :param norm: str 邻接矩阵归一化方式
            'left': S=D^{-1}A, 'right': S=AD^{-1}, 'both': S=D^{-1/2}AD^{-1/2}
        """
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.norm = norm

    @torch.no_grad()
    def forward(self, g, labels, mask=None, post_step=None):
        """
        :param g: DGLGraph 无向图
        :param labels: tensor(N, C) one-hot标签
        :param mask: tensor(N), optional 有标签顶点mask
        :param post_step: callable, optional f: tensor(N, C) -> tensor(N, C)
        :return: tensor(N, C) 预测标签概率
        """
        with g.local_scope():
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]
            else:
                y = labels

            residual = (1 - self.alpha) * y
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5 if self.norm == 'both' else -1).unsqueeze(1)  # (N, 1)
            for _ in range(self.num_layers):
                if self.norm in ('both', 'right'):
                    y *= norm
                g.ndata['h'] = y
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = self.alpha * g.ndata.pop('h')
                if self.norm in ('both', 'left'):
                    y *= norm
                y += residual
                if post_step is not None:
                    y = post_step(y)
            return y


class CorrectAndSmooth(nn.Module):

    def __init__(
            self, num_correct_layers, correct_alpha, correct_norm,
            num_smooth_layers, smooth_alpha, smooth_norm, scale=1.0):
        """C&S模型"""
        super().__init__()
        self.correct_prop = LabelPropagation(num_correct_layers, correct_alpha, correct_norm)
        self.smooth_prop = LabelPropagation(num_smooth_layers, smooth_alpha, smooth_norm)
        self.scale = scale

    def correct(self, g, labels, base_pred, mask):
        """Correct步，修正基础预测中的误差

        :param g: DGLGraph 无向图
        :param labels: tensor(N, C) one-hot标签
        :param base_pred: tensor(N, C) 基础预测
        :param mask: tensor(N) 训练集mask
        :return: tensor(N, C) 修正后的预测
        """
        err = torch.zeros_like(base_pred)  # (N, C)
        err[mask] = labels[mask] - base_pred[mask]

        # FDiff-scale: 对训练集固定误差
        def fix_input(y):
            y[mask] = err[mask]
            return y

        smoothed_err = self.correct_prop(g, err, post_step=fix_input)  # \hat{E}
        corrected_pred = base_pred + self.scale * smoothed_err  # Z^{(r)}
        corrected_pred[corrected_pred.isnan()] = base_pred[corrected_pred.isnan()]
        return corrected_pred

    def smooth(self, g, labels, corrected_pred, mask):
        """Smooth步，平滑最终预测

        :param g: DGLGraph 无向图
        :param labels: tensor(N, C) one-hot标签
        :param corrected_pred: tensor(N, C) 修正后的预测
        :param mask: tensor(N) 训练集mask
        :return: tensor(N, C) 最终预测
        """
        guess = corrected_pred
        guess[mask] = labels[mask]
        return self.smooth_prop(g, guess)

    def forward(self, g, labels, base_pred, mask):
        """
        :param g: DGLGraph 无向图
        :param labels: tensor(N, C) one-hot标签
        :param base_pred: tensor(N, C) 基础预测
        :param mask: tensor(N) 训练集mask
        :return: tensor(N, C) 最终预测
        """
        corrected_pred = self.correct(g, labels, base_pred, mask)
        return self.smooth(g, labels, corrected_pred, mask)
