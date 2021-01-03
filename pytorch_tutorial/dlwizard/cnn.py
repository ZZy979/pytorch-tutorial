import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_tutorial.dlwizard.utils import train_mnist


# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/
class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        """
        :param x: tensor(N, 28, 28) 输入图像
        :return: tensor(N, 10)
        """
        out = F.relu(self.conv1(x))  # (N, 16, 28, 28)
        out = self.pool1(out)  # (N, 16, 14, 14)
        out = F.relu(self.conv2(out))  # (N, 32, 14, 14)
        out = self.pool2(out)  # (N, 32, 7, 7)
        out = out.view(out.shape[0], -1)  # (N, 32 * 7 * 7)
        out = self.fc1(out)  # (N, 10)
        return out


def main():
    model = CNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for p in model.parameters():
        print(p.size())
    train_mnist(model, optimizer, (-1, 1, 28, 28))


if __name__ == '__main__':
    main()
