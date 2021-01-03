# PyTorch示例代码
## beginner - [官方教程](https://pytorch.org/tutorials/)
* two_layer_net.py - [两层全连接网络](https://github.com/pytorch/tutorials/blob/09460601a9f914511d87c12c4e0b04dc21df3086/beginner_source/pytorch_with_examples.rst)
（[原链接](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) 已替换为其他示例）
* neural_networks_tutorial.py - [神经网络示例](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
* cifar10_tutorial.py - [CIFAR10图像分类器](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## dlwizard - [Deep Learning Wizard](https://www.deeplearningwizard.com/deep_learning/intro/)
* linear_regression.py - [线性回归](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/)
* logistic_regression.py - [逻辑回归](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_logistic_regression/)
* fnn.py - [前馈神经网络](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/)
* cnn.py - [卷积神经网络](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/)

## gnn - 图神经网络
### dgl - [DGL官方文档示例](https://docs.dgl.ai/)
* dgl_first_demo.py - [DGL at a Glance](https://docs.dgl.ai/tutorials/basics/1_first.html)
* node_clf.py, node_clf_hetero.py - [顶点分类/回归](https://docs.dgl.ai/guide/training-node.html)
* edge_clf.py, edge_clf_hetero.py, edge_type_hetero.py - [边分类/回归](https://docs.dgl.ai/guide/training-edge.html)
* link_pred.py, link_pred_hetero.py - [连接预测](https://docs.dgl.ai/guide/training-link.html)
* graph_clf.py, graph_clf_hetero.py - [图分类](https://docs.dgl.ai/guide/training-graph.html)

### GNN模型
* gcn - Graph Convolutional Network (GCN)
[论文链接](https://arxiv.org/abs/1609.02907)
| [DGL教程](https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html)
| [DGL实现](https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn)
* rgcn - Relational Graph Convolutional Network (R-GCN)
[论文链接](https://arxiv.org/abs/1703.06103)
| [DGL教程](https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn)
| [DGL实现（异构图）](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero)
