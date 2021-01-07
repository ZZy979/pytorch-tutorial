使用PyCharm运行

实体分类
AIFB
python train_entity_clf.py --dataset=aifb --num-hidden=16 --num-bases=0 --weight-decay=0

MUTAG
python train_entity_clf.py --dataset=mutag --num-hidden=16 --num-bases=30

BGS
python train_entity_clf.py --dataset=bgs --num-hidden=16 --num-bases=40

AM
python train_entity_clf.py --dataset=am --num-hidden=10 --num-bases=40

连接预测
WN18
python train_link_pred.py --dataset=wn18 --num-hidden=200 --num-layers=1 --num-bases=2 --dropout=0.4

FB15k
python train_link_pred.py --dataset=FB15k --num-hidden=200 --num-layers=1 --num-bases=2 --dropout=0.4

FB15k-237
python train_link_pred.py --dataset=FB15k-237 --num-hidden=500 --num-layers=2 --regularizer=bdd --num-bases=100 --dropout=0.4
