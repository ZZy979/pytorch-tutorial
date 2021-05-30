顶点分类
ACM
python -m gnn.hgconv.train --dataset=acm --task=clf --epochs=20
Test Micro-F1 0.9180 | Test Macro-F1 0.9192

IMDb
python -m gnn.hgconv.train --dataset=imdb --task=clf
Test Micro-F1 0.5506 | Test Macro-F1 0.5471

顶点聚类
ACM
python -m gnn.hgconv.train --dataset=acm --task=cluster
Test NMI 0.7099 | Test ARI 0.7600

IMDb
python -m gnn.hgconv.train --dataset=imdb --task=cluster --epochs=10
Test NMI 0.1150 | Test ARI 0.0907
