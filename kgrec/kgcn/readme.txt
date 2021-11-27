MovieLens-20M
python -m kgrec.kgcn.train --dataset=movie --num-hidden=32 --aggregator=sum --num-hops=2 --epochs=5 --batch-size=65536 --neighbor-size=4 --lr=0.02 --weight-decay=0.0000001
Train AUC 0.9742 | Train F1 0.9293 | Test AUC 0.9656 | Test F1 0.9175
作者代码结果：
auc_score: 0.9903, f1_score: 0.9678

Last.FM
python -m kgrec.kgcn.train --dataset=music --num-hidden=16 --aggregator=sum --num-hops=1 --epochs=30 --batch-size=32 --neighbor-size=8 --lr=0.0005 --weight-decay=0.0001
Train AUC 0.9793 | Train F1 0.9245 | Test AUC 0.8057 | Test F1 0.7243
作者代码结果：
auc_score: 0.8590, f1_score: 0.7713

（虽然论文中的公式看起来与普通的消息传递一样，但使用图上的消息传递并不容易实现，因此作者代码使用普通的张量计算实现，这也是必须为每个实体采样固定数量的邻居的原因）
