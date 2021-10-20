## ACM
`python -m gnn.heco.train --dataset=acm`

```
Micro-F1 0.8830 | Macro-F1 0.8850 | AUC 0.9700
NMI 0.5954 | ARI 0.6138
```

作者代码结果：
```
Loading 199th epoch
    [Classification] Macro-F1_mean: 0.8686 var: 0.0068  Micro-F1_mean: 0.8645 var: 0.0064 auc 0.9548
    [Classification] Macro-F1_mean: 0.8649 var: 0.0038  Micro-F1_mean: 0.8652 var: 0.0046 auc 0.9598
    [Classification] Macro-F1_mean: 0.8796 var: 0.0063  Micro-F1_mean: 0.8769 var: 0.0059 auc 0.9543
NMI 0.5634 | ARI 0.5347
```

## DBLP
`python -m gnn.heco.train --dataset=dblp --feat-drop=0.4 --attn-drop=0.35 --tau=0.9`

```
Micro-F1 0.9070 | Macro-F1 0.9032 | AUC 0.9827
NMI 0.6784 | ARI 0.7030
```

作者代码结果：
```
Loading 196th epoch
    [Classification] Macro-F1_mean: 0.9122 var: 0.0032  Micro-F1_mean: 0.9188 var: 0.0033 auc 0.9816
    [Classification] Macro-F1_mean: 0.8970 var: 0.0029  Micro-F1_mean: 0.9001 var: 0.0039 auc 0.9796
    [Classification] Macro-F1_mean: 0.9092 var: 0.0027  Micro-F1_mean: 0.9174 var: 0.0026 auc 0.9844
NMI 0.7086 | ARI 0.7645
```

## Freebase
`python -m gnn.heco.train --dataset=freebase --feat-drop=0.1 --attn-drop=0.3 --tau=0.5 --lr=0.001`

```
Micro-F1 0.5600 | Macro-F1 0.5404 | AUC 0.7307
NMI 0.1773 | ARI 0.2003
```

作者代码结果：
```
Loading 197th epoch
    [Classification] Macro-F1_mean: 0.5493 var: 0.0149  Micro-F1_mean: 0.5739 var: 0.0224 auc 0.7465
    [Classification] Macro-F1_mean: 0.6004 var: 0.0061  Micro-F1_mean: 0.6277 var: 0.0102 auc 0.7709
    [Classification] Macro-F1_mean: 0.5671 var: 0.0223  Micro-F1_mean: 0.5982 var: 0.0290 auc 0.7418
NMI 0.1850 | ARI 0.2121
```

## AMiner
`python -m gnn.heco.train --dataset=aminer --feat-drop=0.5 --attn-drop=0.5 --tau=0.5 --lr=0.003`

```
Micro-F1 0.7350 | Macro-F1 0.6519 | AUC 0.8800
NMI 0.3110 | ARI 0.2773
```

作者代码结果：
```
Loading 199th epoch
    [Classification] Macro-F1_mean: 0.6728 var: 0.0067  Micro-F1_mean: 0.7421 var: 0.0071 auc 0.8743
    [Classification] Macro-F1_mean: 0.6851 var: 0.0032  Micro-F1_mean: 0.7569 var: 0.0034 auc 0.8874
    [Classification] Macro-F1_mean: 0.7232 var: 0.0026  Micro-F1_mean: 0.7927 var: 0.0023 auc 0.9038
NMI 0.3170 | ARI 0.2810
```


## 踩坑记录
一开始自己实现的模型性能比作者提供的代码差很多

花了两天时间逐个部分对比差异，结果发现该模型非常不稳定，对数据、超参数等各种细节非常敏感：

* 基于元路径的邻居图的邻接矩阵是否归一化
* 将作者自己实现的GCN改为dgl的GraphConv、归一化方式由both改为right，性能提升
* 将作者自己实现的GAT改为dgl的GATConv，性能下降（主要区别在于attn_drop的使用方式）
* 参数初始化（使用nn.init.xavier_normal_会提升）
* 正样本采样（np.argsort可以 torch.argsort或torch.topk就不行）

都对结果有很大影响（尤其是顶点聚类）！

经过修改，基本达到与作者代码相近的结果
