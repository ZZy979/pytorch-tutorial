"""参考：<https://docs.dgl.ai/tutorials/basics/1_first.html>"""
import itertools

import dgl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


def build_karate_club_graph():
    src = np.array([
        1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21, 25, 25, 27, 27,
        27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33
    ])
    dst = np.array([
        0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30,
        31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32
    ])
    return dgl.to_bidirected(dgl.graph((src, dst)))


def draw_graph(g):
    nx_g = g.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(nx_g)
    nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()


class GCN(nn.Module):

    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


def draw(i, g, ax, all_logits):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(
        g.to_networkx().to_undirected(), pos, node_color=colors,
        with_labels=True, node_size=300, ax=ax
    )


def main():
    # Step 1: Creating a graph in DGL
    g = build_karate_club_graph()
    print('We have {} nodes.'.format(g.number_of_nodes()))
    print('We have {} edges.'.format(g.number_of_edges()))
    draw_graph(g)

    # Step 2: Assign features to nodes or edges
    embed = nn.Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
    g.ndata['feat'] = embed.weight
    print("node 2's input feature =", g.ndata['feat'][2])

    # Step 3: Define a Graph Convolutional Network (GCN)
    # input feature size of 5 -> hidden size of 5
    # -> output feature size of 2 <=> 2 groups of the karate club
    net = GCN(5, 5, 2)

    # Step 4: Data preparation and initialization
    inputs = embed.weight
    labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
    labels = torch.tensor([0, 1])  # their labels are different

    # Step 5: Train then visualize
    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
    all_logits = []
    for epoch in range(50):
        logits = net(g, inputs)
        # we save the logits for visualization later
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        loss = F.nll_loss(logp[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()
    # 在PyCharm中运行无法显示动画，删掉"ani="也无法显示。。
    ani = animation.FuncAnimation(
        fig, draw, frames=len(all_logits), fargs=(g, ax, all_logits), interval=200
    )
    plt.show()


if __name__ == '__main__':
    main()
