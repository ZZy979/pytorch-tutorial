Cora
python -m gnn.supergat.train --dataset=cora
Test Accuracy 0.8140

CiteSeer
python -m gnn.supergat.train --dataset=citeseer --epochs=80
Test Accuracy 0.6930

PubMed
python -m gnn.supergat.train --dataset=pubmed
Test Accuracy 0.7790

ogbn-arxiv
python -m gnn.supergat.train --dataset=ogbn-arxiv --ogb-root=/home/zzy/ogb/ --num-hidden=16 --dropout=0.2 --lr=0.05
Test Accuracy 0.1101

Cora-Full
python -m gnn.supergat.train --dataset=cora_full --lr=0.01
Test Accuracy 0.5175

CS
python -m gnn.supergat.train --dataset=cs --lr=0.01
Test Accuracy 0.9162

Physics
python -m gnn.supergat.train --dataset=physics --lr=0.01
Test Accuracy 0.9555

Photo
python -m gnn.supergat.train --dataset=photo --lr=0.01
Test Accuracy 0.9255

Computers
python -m gnn.supergat.train --dataset=computers --lr=0.01
Test Accuracy 0.8906
