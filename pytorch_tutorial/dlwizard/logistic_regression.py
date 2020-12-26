import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST


# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def main():
    train_dataset = MNIST('./data', True, transforms.ToTensor(), download=True)
    test_dataset = MNIST('./data', False, transforms.ToTensor())
    batch_size = 100
    n_iters = 3000
    n_epochs = int(n_iters / (len(train_dataset) / batch_size))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = LogisticRegressionModel(28 * 28, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    for p in model.parameters():
        print(p.size())

    iteration = 0
    for epoch in range(n_epochs):
        for images, labels in train_loader:
            images = images.view(-1, 28 * 28).requires_grad_()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 500 == 0:
                # Calculate Accuracy
                correct = 0
                for test_images, test_labels in test_loader:
                    test_images = test_images.view(-1, 28 * 28).requires_grad_()
                    outputs = model(test_images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == test_labels).sum().item()
                accuracy = correct / len(test_dataset)
                print('Iteration: {}. Loss: {}. Accuracy: {:.2%}'.format(
                    iteration, loss.item(), accuracy
                ))


if __name__ == '__main__':
    main()
