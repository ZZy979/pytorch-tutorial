# PyTorch示例代码

## beginner - [PyTorch官方教程](https://pytorch.org/tutorials/)
* two_layer_net.py - [两层全连接网络](https://github.com/pytorch/tutorials/blob/09460601a9f914511d87c12c4e0b04dc21df3086/beginner_source/pytorch_with_examples.rst)
（[原链接](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) 已替换为其他示例）
* neural_networks_tutorial.py - [神经网络示例](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
* cifar10_tutorial.py - [CIFAR10图像分类器](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## dlwizard - [Deep Learning Wizard](https://www.deeplearningwizard.com/deep_learning/intro/)
* linear_regression.py - [线性回归](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/)
* logistic_regression.py - [逻辑回归](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_logistic_regression/)
* fnn.py - [前馈神经网络](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/)
* cnn.py - [卷积神经网络](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/)
* rnn.py - [循环神经网络](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/)
* lstm.py - [LSTM](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/)

## gnn - 图神经网络
复现的GNN模型全部使用DGL实现，部分模型参考了[DGL官方示例](https://github.com/dmlc/dgl/tree/master/examples)

运行方式：使用Python的`-m`参数

例如：要运行gnn/gcn/train.py则执行`python -m gnn.gcn.train`

### dgl - [DGL官方文档示例](https://docs.dgl.ai/)
[训练GNN模型](https://docs.dgl.ai/en/latest/guide/training.html)
* dgl_first_demo.py - [DGL at a Glance](https://docs.dgl.ai/en/0.5.x/tutorials/basics/1_first.html)
* node_clf.py, node_clf_hetero.py - [顶点分类/回归](https://docs.dgl.ai/en/latest/guide/training-node.html)
* edge_clf.py, edge_clf_hetero.py, edge_type_hetero.py - [边分类/回归](https://docs.dgl.ai/en/latest/guide/training-edge.html)
* link_pred.py, link_pred_hetero.py - [连接预测](https://docs.dgl.ai/en/latest/guide/training-link.html)
* graph_clf.py, graph_clf_hetero.py - [图分类](https://docs.dgl.ai/en/latest/guide/training-graph.html)

[大图上的随机训练](https://docs.dgl.ai/en/latest/guide/minibatch.html)
* node_clf_mb.py, node_clf_hetero_mb.py - [使用邻居采样的顶点分类](https://docs.dgl.ai/en/latest/guide/minibatch-node.html)
* node_clf_mb.py, edge_clf_hetero_mb.py - [使用邻居采样的边分类](https://docs.dgl.ai/en/latest/guide/minibatch-edge.html)
* link_pred_mb.py, link_pred_hetero_mb.py - [使用邻居采样的连接预测](https://docs.dgl.ai/en/latest/guide/minibatch-link.html)

### GNN模型
* gcn - Graph Convolutional Network (GCN)
[论文链接](https://arxiv.org/abs/1609.02907)
| [官方代码](https://github.com/tkipf/gcn)
| [DGL实现](https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn)
* gat - Graph Attention Networks (GAT)
[论文链接](https://arxiv.org/abs/1710.10903)
| [官方代码](https://github.com/PetarV-/GAT)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat)
* rgcn - Relational Graph Convolutional Network (R-GCN)
[论文链接](https://arxiv.org/abs/1703.06103)
| [官方代码](https://github.com/tkipf/relational-gcn)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn)
| [DGL实现（异构图）](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero)
* hetgnn - Heterogeneous Graph Neural Network (HetGNN)
[论文链接](https://dl.acm.org/doi/pdf/10.1145/3292500.3330961)
| [官方代码](https://github.com/chuxuzhang/KDD2019_HetGNN)
* han - Heterogeneous Graph Attention Network (HAN)
[论文链接](https://arxiv.org/abs/1903.07293)
| [官方代码](https://github.com/Jhy1993/HAN)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/han)
* hgt - Heterogeneous Graph Transformer (HGT)
[论文链接](https://arxiv.org/pdf/2003.01332)
| [官方代码](https://github.com/acbull/pyHGT)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/hgt)
* magnn - Metapath Aggregated Graph Neural Network (MAGNN)
[论文链接](https://arxiv.org/pdf/2002.01680)
| [官方代码](https://github.com/cynricfu/MAGNN)
* sign - SIGN: Scalable Inception Graph Neural Networks (SIGN)
[论文链接](https://arxiv.org/pdf/2004.11198)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/sign)
* hgconv - Hybrid Micro/Macro Level Convolution for Heterogeneous Graph Learning (HGConv)
[论文链接](https://arxiv.org/pdf/2012.14722)
| [官方代码](https://github.com/yule-BUAA/HGConv)
* supergat - How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision (SuperGAT)
[论文链接](https://openreview.net/pdf?id=Wi5KUNlqWty)
| [官方代码](https://github.com/dongkwan-kim/SuperGAT)
* metapath2vec - metapath2vec: Scalable Representation Learning for Heterogeneous Networks
[论文链接](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)
| [官方代码](https://ericdongyx.github.io/metapath2vec/m2v.html)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/metapath2vec)
* rhgnn - Heterogeneous Graph Representation Learning with Relation Awareness (R-HGNN)
[论文链接](https://arxiv.org/pdf/2105.11122)
| [官方代码](https://github.com/yule-BUAA/R-HGNN/)
* lp - Label Propagation
[论文链接](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/label_propagation)
* cs - Correct and Smooth (C&S)
[论文链接](https://arxiv.org/pdf/2010.13993)
| [官方代码](https://github.com/CUAI/CorrectAndSmooth)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/correct_and_smooth)
* heco - Self-Supervised Heterogeneous Graph Neural Network with Co-Contrastive Learning (HeCo)
[论文链接](https://arxiv.org/pdf/2105.09111)
| [官方代码](https://github.com/liun-online/HeCo)

## kgrec - 基于知识图谱的推荐算法
* kgcn - Knowledge Graph Convolutional Networks for Recommender Systems (KGCN)
[论文链接](https://arxiv.org/pdf/1904.12575)
| [官方代码](https://github.com/hwwang55/KGCN)
| [PyTorch实现](https://github.com/zzaebok/KGCN-pytorch)

## nlp - 自然语言处理
### tfms - [transformers官方示例](https://huggingface.co/transformers/)
* seq_clf_pipeline.py, seq_clf_model.py - [语义分析](https://huggingface.co/transformers/task_summary.html#sequence-classification)
* eqa_pipeline.py, eqa_model.py - [提取式问答](https://huggingface.co/transformers/task_summary.html#extractive-question-answering)
* masked_lm_pipeline.py, masked_lm_model.py - [屏蔽语言建模](https://huggingface.co/transformers/task_summary.html#masked-language-modeling)
* causal_lm_model.py - [因果语言建模](https://huggingface.co/transformers/task_summary.html#causal-language-modeling)
* text_gen_pipeline.py, text_gen_model.py - [文本生成](https://huggingface.co/transformers/task_summary.html#text-generation)
