# Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks
This repository is the implementation of the model COLA from paper: **Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks**.

## Requirements
```
python=3.8
torch=1.13.0
pyg=2.3.0
PyTorch Lightning=2.0.1.post0
ogb=1.3.6
pygcl=0.1.2
wandb=0.14.2
ruamel.yaml=0.17.21
```

## Usages
Using following command the run the code. 
Here is the example of running a 2-way 5-shot task on CiteSeer dataset.
```
python main.py --dataset=CiteSeer --n_way=2 --k_shot=5
```
