顶点分类
ACM
python -m gnn.han.train --dataset=acm --hetero --task=clf
Test Micro-F1 0.9091 | Test Macro-F1 0.9114

DBLP
python -m gnn.han.train --dataset=dblp --hetero --task=clf
Test Micro-F1 0.9247 | Test Macro-F1 0.9186

IMDb
python -m gnn.han.train --dataset=imdb --hetero --task=clf
Test Micro-F1 0.5650 | Test Macro-F1 0.5623

顶点聚类
ACM
python -m gnn.han.train --dataset=acm --hetero --task=cluster
Test NMI 0.7090 | Test ARI 0.7541

DBLP
python -m gnn.han.train --dataset=dblp --hetero --task=cluster
Test NMI 0.7688 | Test ARI 0.8271

IMDb
python -m gnn.han.train --dataset=imdb --hetero --task=cluster
Test NMI 0.1135 | Test ARI 0.1271
