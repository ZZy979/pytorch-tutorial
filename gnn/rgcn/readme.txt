实体分类
AIFB
python -m gnn.rgcn.train_entity_clf --dataset=aifb --num-hidden=16 --num-bases=0 --weight-decay=0
Test Accuracy 0.9167

MUTAG
python -m gnn.rgcn.train_entity_clf --dataset=mutag --num-hidden=16 --num-bases=30
Test Accuracy 0.7647

BGS
python -m gnn.rgcn.train_entity_clf --dataset=bgs --num-hidden=16 --num-bases=40
Test Accuracy 0.8966

AM
python -m gnn.rgcn.train_entity_clf --dataset=am --num-hidden=10 --num-bases=40
Test Accuracy 0.8434

连接预测
WN18
python -m gnn.rgcn.train_link_pred --dataset=wn18 --num-hidden=200 --num-layers=1 --num-bases=2 --dropout=0.4

FB15k
python -m gnn.rgcn.train_link_pred --dataset=FB15k --num-hidden=200 --num-layers=1 --num-bases=2 --dropout=0.4

FB15k-237
python -m gnn.rgcn.train_link_pred --dataset=FB15k-237 --num-hidden=500 --num-layers=2 --regularizer=bdd --num-bases=100 --dropout=0.4
