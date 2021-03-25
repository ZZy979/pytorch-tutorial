# PyTorch示例代码
## 运行方式
使用Python的`-m`参数

例如：要运行gnn/gcn/train.py则执行 `python -m gnn.gcn.train`

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
### dgl - [DGL官方文档示例](https://docs.dgl.ai/)
* dgl_first_demo.py - [DGL at a Glance](https://docs.dgl.ai/tutorials/basics/1_first.html)
* node_clf.py, node_clf_hetero.py - [顶点分类/回归](https://docs.dgl.ai/guide/training-node.html)
* edge_clf.py, edge_clf_hetero.py, edge_type_hetero.py - [边分类/回归](https://docs.dgl.ai/guide/training-edge.html)
* link_pred.py, link_pred_hetero.py - [连接预测](https://docs.dgl.ai/guide/training-link.html)
* graph_clf.py, graph_clf_hetero.py - [图分类](https://docs.dgl.ai/guide/training-graph.html)
* node_clf_mb.py, node_clf_hetero_mb.py - [使用邻居采样的顶点分类](https://docs.dgl.ai/guide/minibatch-node.html)

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
* gat - Graph Attention Networks (GAT)
[论文链接](https://arxiv.org/abs/1710.10903)
| [DGL教程](https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat)
* han - Heterogeneous Graph Attention Network (HAN)
[论文链接](https://arxiv.org/abs/1903.07293)
| [官方代码](https://github.com/Jhy1993/HAN)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/han)
* magnn - Metapath Aggregated Graph Neural Network (MAGNN)
[论文链接](https://arxiv.org/pdf/2002.01680)
| [官方代码](https://github.com/cynricfu/MAGNN)
* sign - SIGN: Scalable Inception Graph Neural Networks (SIGN)
[论文链接](https://arxiv.org/pdf/2004.11198)
| [DGL实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/sign)
* hgconv - Hybrid Micro/Macro Level Convolution for Heterogeneous Graph Learning (HGConv)
[论文链接](https://arxiv.org/pdf/2012.14722)
| [官方代码](https://github.com/yule-BUAA/HGConv)

## nlp - 自然语言处理
### tfms - [transformers官方示例](https://huggingface.co/transformers/)
* seq_clf_pipeline.py, seq_clf_model.py - [语义分析](https://huggingface.co/transformers/task_summary.html#sequence-classification)
* eqa_pipeline.py, eqa_model.py - [提取式问答](https://huggingface.co/transformers/task_summary.html#extractive-question-answering)
* masked_lm_pipeline.py, masked_lm_model.py - [屏蔽语言建模](https://huggingface.co/transformers/task_summary.html#masked-language-modeling)
* causal_lm_model.py - [因果语言建模](https://huggingface.co/transformers/task_summary.html#causal-language-modeling)
* text_gen_pipeline.py, text_gen_model.py - [文本生成](https://huggingface.co/transformers/task_summary.html#text-generation)
