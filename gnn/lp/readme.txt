Cora
python -m gnn.lp.train --dataset=cora
Test Accuracy 0.6920

Citeseer
python -m gnn.lp.train --dataset=citeseer --num-layers=100 --alpha=0.99
Test Accuracy 0.5130

Pubmed
python -m gnn.lp.train --dataset=pubmed --num-layers=60 --alpha=1
Test Accuracy 0.7140
