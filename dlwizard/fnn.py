import torch.nn as nn
import torch.optim as optim

from dlwizard.common import train_mnist


# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
class FeedforwardNeuralNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out


def main():
    model = FeedforwardNeuralNetwork(28 * 28, 100, 10, nn.ReLU())
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for p in model.parameters():
        print(p.size())
    train_mnist(model, optimizer, (-1, 28 * 28))


if __name__ == '__main__':
    main()
