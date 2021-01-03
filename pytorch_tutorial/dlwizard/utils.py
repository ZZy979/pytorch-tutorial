import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST


def train_mnist(model, optimizer, view_shape=(-1, 28, 28)):
    train_dataset = MNIST('D:\\torchdata', True, transforms.ToTensor(), download=True)
    test_dataset = MNIST('D:\\torchdata', False, transforms.ToTensor())
    batch_size = 100
    n_epochs = 5
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    iteration = 0
    for epoch in range(n_epochs):
        for images, labels in train_loader:
            images = images.view(view_shape).requires_grad_()
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
                    test_images = test_images.view(view_shape).requires_grad_()
                    outputs = model(test_images)
                    predicted = torch.argmax(outputs.data, dim=1)
                    correct += (predicted == test_labels).sum().item()
                accuracy = correct / len(test_dataset)
                print('Iteration: {}. Loss: {}. Accuracy: {:.2%}'.format(
                    iteration, loss.item(), accuracy
                ))
