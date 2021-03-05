顶点分类
ACM
python -m gnn.hgconv.train --dataset=acm --task=clf
Test Micro-F1 0.9247 | Test Macro-F1 0.9253

IMDb
python -m gnn.hgconv.train --dataset=imdb --task=clf
Test Micro-F1 0.5512 | Test Macro-F1 0.5506

顶点聚类
ACM
python -m gnn.hgconv.train --dataset=acm --task=cluster
Test NMI 0.7294 | Test ARI 0.7742

IMDb
python -m gnn.hgconv.train --dataset=imdb --task=cluster
Test NMI 0.1021 | Test ARI 0.1055
