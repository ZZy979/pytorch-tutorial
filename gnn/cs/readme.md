# Cora
## Linear + C&S
`python -m gnn.cs.train --dataset=cora --base-model=Linear --num-hidden=64 --epochs=10`
```
Base model Linear
Test Acc 0.4710
C&S
Test Acc 0.8010
```

## MLP + C&S
`python -m gnn.cs.train --dataset=cora --base-model=MLP --num-hidden=64 --epochs=10`
```
Base model MLP
Test Acc 0.3190
C&S
Test Acc 0.7960
```

# Citeseer
## Linear + C&S
`python -m gnn.cs.train --dataset=citeseer --base-model=Linear --num-hidden=64 --epochs=10`
```
Base model Linear
Test Acc 0.4710
C&S
Test Acc 0.6650
```

## MLP + C&S
`python -m gnn.cs.train --dataset=citeseer --base-model=MLP --num-hidden=64 --epochs=10`
```
Base model MLP
Test Acc 0.2310
C&S
Test Acc 0.6110
```

# Pubmed
## Linear + C&S
`python -m gnn.cs.train --dataset=pubmed --base-model=Linear --num-hidden=64 --epochs=10`
```
Base model Linear
Test Acc 0.6870
C&S
Test Acc 0.7780
```

## MLP + C&S
`python -m gnn.cs.train --dataset=pubmed --base-model=MLP --num-hidden=64 --epochs=10`
```
Base model MLP
Test Acc 0.4350
C&S
Test Acc 0.7380
```

# ogbn-products
## Linear + C&S
`python -m gnn.cs.train --dataset=ogbn-products --ogb-root=/home/zzy/ogb --base-model=Linear --correct-alpha=0.6 --smooth-alpha=0.9`
```
Base model Linear
Test Acc 0.4777
C&S
Test Acc 0.7220
```

# ogbn-arxiv
## Linear + C&S
`python -m gnn.cs.train --dataset=ogbn-arxiv --ogb-root=/home/zzy/ogb --base-model=Linear --correct-alpha=0.8 --correct-norm=right --smooth-alpha=0.6`
```
Base model Linear
Test Acc 0.5245
C&S
Test Acc 0.6776
```

## MLP + C&S
`python -m gnn.cs.train --dataset=ogbn-arxiv --ogb-root=/home/zzy/ogb --base-model=MLP --correct-alpha=0.979 --correct-norm=left --smooth-alpha=0.756 --smooth-norm=right`
```
Base model MLP
Test Acc 0.5616
C&S
Test Acc 0.6717
```
