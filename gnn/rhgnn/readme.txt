ogbn-mag数据集

1.预训练顶点嵌入
见https://github.com/ZZy979/GNN-Recommendation/blob/main/gnnrec/hge/readme.md#%E9%A2%84%E8%AE%AD%E7%BB%83%E9%A1%B6%E7%82%B9%E5%B5%8C%E5%85%A5

2.训练模型
python -m gnn.rhgnn.train /home/zzy/ogb/ /home/zzy/GNN-Recommendation/model/word2vec/ogbn-mag.model
Test Acc 0.5201
