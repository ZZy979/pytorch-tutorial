import argparse

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import PPIDataset
from sklearn.metrics import f1_score
from torch.utils.data.dataloader import DataLoader

from pytorch_tutorial.gnn.gat.model import GAT


def evaluate(model, g, feats, labels):
    with torch.no_grad():
        model.eval()
        logits = model(g, feats)
        predict = np.where(logits.data.numpy() >= 0., 1, 0)
        return f1_score(labels.data.numpy(), predict, average='micro')


def train(args):
    train_dataset = PPIDataset('train')
    valid_dataset = PPIDataset('valid')
    test_dataset = PPIDataset('test')
    train_dataloader = DataLoader(train_dataset, args.batch_size, collate_fn=dgl.batch)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, collate_fn=dgl.batch)
    test_dataloader = DataLoader(test_dataset, args.batch_size, collate_fn=dgl.batch)

    num_feats = train_dataset[0].ndata['feat'].shape[1]
    num_classes = train_dataset.num_labels

    model = GAT(num_feats, 256, num_classes, [4, 4, 6])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for bg in train_dataloader:
            logits = model(bg, bg.ndata['feat'])
            loss = F.binary_cross_entropy_with_logits(logits, bg.ndata['label'].float())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch {:04d} | Loss {:.4f}'.format(epoch, np.array(losses).mean()))
        if epoch % 5 == 0:
            val_scores = [
                evaluate(model, bg, bg.ndata['feat'], bg.ndata['label']) for bg in valid_dataloader
            ]
            print('Val F1-score {:.4f}'.format(np.array(val_scores).mean()))

    print()
    test_scores = [
        evaluate(model, bg, bg.ndata['feat'], bg.ndata['label']) for bg in test_dataloader
    ]
    print('Test F1-score {:.4f}'.format(np.array(test_scores).mean()))


def main():
    parser = argparse.ArgumentParser(description='GAT Inductive Training')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0., help='weight decay')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
