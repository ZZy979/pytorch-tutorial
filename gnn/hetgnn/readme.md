# 运行代码
## 1.随机游走
`python -m gnn.hetgnn.random_walk D:\dgldata\aminer2\corpus.txt`

## 2.预训练顶点嵌入
`python -m gnn.metapath2vec.train_word2vec --size=128 D:\dgldata\aminer2\corpus.txt D:\dgldata\aminer2\aminer2.model`

## 3.预处理，生成邻居图和输入特征
`python -m gnn.hetgnn.preprocess D:\dgldata\aminer2\corpus.txt D:\dgldata\aminer2\aminer2.model D:\dgldata\aminer2\`

## 4.训练模型（无监督），学习顶点嵌入
`python -m gnn.hetgnn.train D:\dgldata\aminer2\ D:\dgldata\aminer2\node_embed.pkl`

## 5.评估顶点嵌入
### 顶点分类
`python -m gnn.hetgnn.eval --task=clf D:\dgldata\aminer2\node_embed.pkl`
```
Train size 10% | Test Micro-F1 0.9588 | Test Macro-F1 0.9573
Train size 30% | Test Micro-F1 0.9657 | Test Macro-F1 0.9647
```

### 顶点聚类
`python -m gnn.hetgnn.eval --task=cluster D:\dgldata\aminer2\node_embed.pkl`
```
NMI 0.8099 | ARI 0.8611
```

# 数据集
作者提供的数据集（只有论文中的Academic II数据集）：
* https://github.com/chuxuzhang/KDD2019_HetGNN/tree/master/data/academic
* https://drive.google.com/file/d/1N6GWsniacaT-L0GPXpi1D3gM2LVOih-A/view?usp=sharing

二者基本相同，但GitHub上的数据包含随机游走结果、生成的训练样本等，都没有原始数据

## 图结构
* 顶点：28646 author, 21044 paper, 18 venue （学者顶点没有0）
* 边：A-P, P-P, P-V

## 文件说明
### A-P边
* a_p_list_train.txt, a_p_list_test.txt 格式：aid:pid,pid,...
* p_a_list_train.txt, p_a_list_test.txt 格式：pid:aid,aid,... （反向边，与上面两个文件完全相同）

### P-P边
* p_p_cite_list_train.txt, p_p_cite_list_test.txt 格式：pid:pid,pid,...
* p_p_citation_list.txt不是两者的并集，无用

### P-V边
* p_v.txt  格式：pid,vid （完整，0<=pid<=21043, 0<=vid<=17）
* v_p_list_train.txt 格式：vid:pid,pid,... （不完整）

### author标签
a_class_train.txt, a_class_test.txt  格式：aid,class （不完整，11360个aid, 0<=class<=3）

### paper特征
* p_title_embed.txt 标题词向量，格式：（第二行开始）pid x1 ... x128 （完整，0<=pid<=21043）
* p_abstract_embed.txt 摘要词向量，格式同上
* p_time.txt 论文年份，格式：pid\ttime （完整，time是论文年份-2005，见作者代码raw_data_process.py）

### 随机游走结果
* het_random_walk.txt 格式：{a|p|v}_{id} × n （不完整，出现的顶点集合同node_net_embedding.txt）
* het_neigh_train.txt 格式：{a|p|v}\_{id}:{a|p|v}\_{id},... （不完整，出现的顶点集合同node_net_embedding.txt）

### 顶点嵌入
node_net_embedding.txt 随机游走结果+word2vec 格式：{a|p|v}_{id} x1 ... x128
（不完整，数量：22821 author, 15930 paper, 18 venue）

### 生成的训练样本
* A-A: a_a_list_train.txt, a_a_list_test.txt 格式：aid, aid, {0|1}  （0/1的含义未知）
* A-P: a_p_cite_list_train.txt, a_p_cite_list_test.txt 格式：aid, pid, {0|1}  （0/1的含义未知）
* A-V: a_v_list_train.txt, a_v_list_test.txt 格式：aid:vid,vid,...,

# 作者代码
作者提供的代码：https://github.com/chuxuzhang/KDD2019_HetGNN/tree/master/code

（~~又臭又长~~）
* raw_data_process.py 将原始数据处理成提供的格式（没有原始数据，因此无用）
* input_data_process.py 随机游走和生成三种训练样本边(A-A, A-P, A-V)
    * 依次从每一个学者顶点出发，游走固定长度L=30，每个顶点重复n=10次
    * 随机游走策略：如果当前顶点为A或V则随机选择一个P邻居；如果当前顶点为P则从A+P+V邻居中随机选择一个
* DeepWalk.py 随机游走结果+word2vec → 预训练的顶点嵌入(node_net_embedding.txt)
* data_generator.py 读取原始数据
    * paper顶点特征
        * p_title_embed 预训练的标题词向量
        * p_abstract_embed 预训练的摘要词向量
        * p_net_embed 预训练的顶点嵌入
        * p_v_net_embed 论文所属期刊的v_net_embed
        * p_a_net_embed 论文作者的a_net_embed的平均
        * p_ref_net_embed 论文引用的论文的p_net_embed的平均
    * author顶点特征
        * a_net_embed 预训练的顶点嵌入
        * a_text_embed 3篇论文的p_abstract_embed的拼接（不足3篇的用最后一篇补齐）
    * venue顶点特征
        * v_net_embed 预训练的顶点嵌入
        * v_text_embed 5篇论文的p_abstract_embed的拼接（不足5篇的用最后一篇补齐）
    * het_walk_restart() 随机游走生成邻居(het_neigh_train.txt)
        * 每种顶点都选择邻居中出现次数最多的10个author、10个paper、3个venue
        * input_data_process.py中的随机游走是为了预训练顶点嵌入生成语料库，与这里的随机游走不同
    * 负采样：sample_het_walk_triple()
* args.py 解析命令行参数
* tools.py 核心模型HetAgg（你管这叫tools？）
* HetGNN.py 整体模型+训练
