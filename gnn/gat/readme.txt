直推式
Cora
python -m gnn.gat.train_transductive --dataset=cora
Test Accuracy 0.8230

Citeseer
python -m gnn.gat.train_transductive --dataset=citeseer
Test Accuracy 0.7010

Pubmed
python -m gnn.gat.train_transductive --dataset=pubmed --num-out-heads=8 --lr=0.01 --weight-decay=0.001
Test Accuracy 0.7950

归纳式
PPI
python -m gnn.gat.train_inductive
Test F1-score 0.9863
