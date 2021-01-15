import torch.nn as nn


class MetapathInstanceEncoder(nn.Module):
    """元路径实例编码器，将一个元路径实例所有中间顶点的特征编码为一个向量。"""

    def forward(self, feat):
        """
        :param feat: tensor(E, L, d_in)
        :return: tensor(E, d_out)
        """
        raise NotImplementedError


class MeanEncoder(MetapathInstanceEncoder):

    def __init__(self, in_dim, out_dim):
        super().__init__()

    def forward(self, feat):
        return feat.mean(dim=1)


class LinearEncoder(MetapathInstanceEncoder):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, feat):
        return self.fc(feat.mean(dim=1))


ENCODERS = {
    'mean': MeanEncoder,
    'linear': LinearEncoder
}


def get_encoder(name, in_dim, out_dim):
    if name in ENCODERS:
        return ENCODERS[name](in_dim, out_dim)
    else:
        raise ValueError('非法编码器名称{}，可选项为{}'.format(name, list(ENCODERS.keys())))
