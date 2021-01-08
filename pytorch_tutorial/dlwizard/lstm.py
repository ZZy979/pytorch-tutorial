import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_tutorial.dlwizard.utils import train_mnist


# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
class LSTM(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        :param x: tensor(batch, seq_len, d_in)
        :return: tensor(batch, d_out)
        """
        # 由于设置了batch_first=True，因此x是(batch, seq_len, d_in)而不是(seq_len, batch, d_in)
        # Initialize hidden state with zeros (num_layers, batch, d_hid)
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()

        # Initialize cell state (num_layers, batch, d_hid)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # Otherwise we'll backprop all the way to the start even after going through another batch
        # out: (batch, seq_len, d_hid), hn, cn: (num_layers, batch, d_hid)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out[:, -1, :] -> (batch, d_hid) --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])  # (batch, d_out)
        return out


def main():
    in_dim = 28
    seq_len = 28
    model = LSTM(in_dim, 100, 1, 10)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for p in model.parameters():
        print(p.size())
    train_mnist(model, optimizer, (-1, seq_len, in_dim))


if __name__ == '__main__':
    main()
