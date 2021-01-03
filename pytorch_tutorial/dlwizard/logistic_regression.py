import torch.nn as nn
import torch.optim as optim

from pytorch_tutorial.dlwizard.utils import train_mnist


# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_logistic_regression/
class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def main():
    model = LogisticRegressionModel(28 * 28, 10)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    for p in model.parameters():
        print(p.size())
    train_mnist(model, optimizer, (-1, 28 * 28))


if __name__ == '__main__':
    main()
