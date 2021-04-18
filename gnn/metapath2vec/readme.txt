1.随机游走生成语料库
python -m gnn.metapath2vec.random_walk --num-walks=1000 --walk-length=100 D:\dgldata\aminer-cs\corpus.txt

2.训练word2vec
python -m gnn.metapath2vec.train_word2vec --size=128 --workers=8 D:\dgldata\aminer-cs\corpus.txt D:\dgldata\aminer-cs\aminer-cs.model

3.评价顶点嵌入
期刊（会议）顶点分类
python -m gnn.metapath2vec.train --ntype=conf --task=clf D:\dgldata\aminer-cs\aminer-cs.model
Train size 5% | Test Micro-F1 0.4453 | Test Macro-F1 0.3634
Train size 10% | Test Micro-F1 0.5620 | Test Macro-F1 0.4602
Train size 20% | Test Micro-F1 0.6667 | Test Macro-F1 0.6281
Train size 30% | Test Micro-F1 0.9043 | Test Macro-F1 0.9096
Train size 40% | Test Micro-F1 0.9259 | Test Macro-F1 0.9273
Train size 50% | Test Micro-F1 0.9552 | Test Macro-F1 0.9563
Train size 60% | Test Micro-F1 0.9815 | Test Macro-F1 0.9838
Train size 70% | Test Micro-F1 1.0000 | Test Macro-F1 1.0000
Train size 80% | Test Micro-F1 1.0000 | Test Macro-F1 1.0000
Train size 90% | Test Micro-F1 1.0000 | Test Macro-F1 1.0000

学者顶点分类
python -m gnn.metapath2vec.train --ntype=author --task=clf D:\dgldata\aminer-cs\aminer-cs.model
Train size 5% | Test Micro-F1 0.9334 | Test Macro-F1 0.9274
Train size 10% | Test Micro-F1 0.9395 | Test Macro-F1 0.9343
Train size 20% | Test Micro-F1 0.9418 | Test Macro-F1 0.9368
Train size 30% | Test Micro-F1 0.9426 | Test Macro-F1 0.9380
Train size 40% | Test Micro-F1 0.9429 | Test Macro-F1 0.9385
Train size 50% | Test Micro-F1 0.9433 | Test Macro-F1 0.9389
Train size 60% | Test Micro-F1 0.9433 | Test Macro-F1 0.9387
Train size 70% | Test Micro-F1 0.9425 | Test Macro-F1 0.9381
Train size 80% | Test Micro-F1 0.9427 | Test Macro-F1 0.9385
Train size 90% | Test Micro-F1 0.9414 | Test Macro-F1 0.9371

期刊（会议）顶点聚类
python -m gnn.metapath2vec.train --ntype=conf --task=cluster D:\dgldata\aminer-cs\aminer-cs.model
Average NMI 0.9173

学者顶点聚类
python -m gnn.metapath2vec.train --ntype=author --task=cluster D:\dgldata\aminer-cs\aminer-cs.model
Average NMI 0.7497


注意：原论文提供的随机游走代码py4genMetaPaths.py中预先计算出学者-期刊对应关系，每一步随机游走是从对应关系中等概率选择
这与dgl.sampling.random_walk()选择顶点的概率并不相等！

例如，有以下异构图
g = dgl.heterograph({
    ('author', 'ap', 'paper'): ([0, 0, 1, 1, 2], [0, 1, 1, 2, 2]),
    ('paper', 'pa', 'author'): ([0, 1, 1, 2, 2], [0, 0, 1, 1, 2]),
    ('paper', 'pc', 'conf'): ([0, 1, 2], [0, 0, 1]),
    ('conf', 'cp', 'paper'): ([0, 0, 1], [0, 1, 2])
})

a0 - p0
   \    \
a1 - p1 - c0
   \
a2 - p2 - c1

（1）原论文代码：计算学者-期刊对应关系
conf_author = {c0: [a0, a0, a1], c1: [a1, a2]}
author_conf = {a0: [c0, c0], a1: [c0, c1], a2: [c1]}
c0 -> 2/3a0 + 1/3a1

（2）dgl.sampling.random_walk(g, [0], metapath=['cp', 'pa'])
c0 -> 1/2p0 + 1/2p1 -> 1/2a0 + 1/4a0 + 1/4a1 = 3/4a0 + 1/4a1

>>> traces, _ = dgl.sampling.random_walk(g, [0] * 1000, metapath=['cp', 'pa'])
>>> authors = traces[:, 2]
>>> authors.unique(return_counts=True)
(tensor([0, 1]), tensor([754, 246]))

解决方法：使用prob参数
>>> cp = g.adj(etype='cp', scipy_fmt='csr')
>>> pa = g.adj(etype='pa', scipy_fmt='csr')
>>> ca = cp * pa
>>> ca.todense()
matrix([[2, 1, 0],
        [0, 1, 1]], dtype=int64)
>>> cag = dgl.heterograph({('conf', 'ca', 'author'): ca.nonzero()})
>>> cag.edata['p'] = torch.from_numpy(ca.data).float()
>>> traces, _ = dgl.sampling.random_walk(cag, [0] * 1000, metapath=['ca'], prob='p')
>>> traces[:, 1].unique(return_counts=True)
(tensor([0, 1]), tensor([670, 330]))
