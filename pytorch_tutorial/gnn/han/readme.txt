顶点分类
ACM
python train.py --dataset=acm --hetero --task=clf
Test Micro-F1 0.9091 | Test Macro-F1 0.9114

DBLP
python train.py --dataset=dblp --hetero --task=clf
Test Micro-F1 0.9247 | Test Macro-F1 0.9186

顶点聚类
ACM
python train.py --dataset=acm --hetero --task=cluster
Test NMI 0.7090 | Test ARI 0.7541

DBLP
python train.py --dataset=dblp --hetero --task=cluster
Test NMI 0.7688 | Test ARI 0.8271
