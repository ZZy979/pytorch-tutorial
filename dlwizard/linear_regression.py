import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/
class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def main():
    X_train = np.arange(11, dtype=np.float32).reshape((-1, 1))
    y_train = 2 * X_train + 1
    model = LinearRegressionModel(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 100
    for epoch in range(epochs):
        # Convert numpy array to torch tensor
        inputs = torch.from_numpy(X_train).requires_grad_()
        labels = torch.from_numpy(y_train)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward to get output
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

    # Purely inference
    predicted = model(torch.from_numpy(X_train).requires_grad_()).data.numpy()
    print(predicted)

    # Clear figure
    plt.clf()

    # Plot true data
    plt.plot(X_train, y_train, 'go', label='True data', alpha=0.5)

    # Plot predictions
    plt.plot(X_train, predicted, '--', label='Predictions', alpha=0.5)

    # Legend and plot
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
