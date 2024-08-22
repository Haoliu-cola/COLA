# Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks
This repository is the implementation of the model COLA from paper: [**Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks**](https://dl.acm.org/doi/abs/10.1145/3589334.3645367).

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
Please use the following command to run the code. 
Here is an example of running a 2-way 5-shot task on the CiteSeer dataset.
```
python main.py --dataset=CiteSeer --n_way=2 --k_shot=5
```

## Citation
```
@inproceedings{liu2024graph,
  title={Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks},
  author={Liu, Hao and Feng, Jiarui and Kong, Lecheng and Tao, Dacheng and Chen, Yixin and Zhang, Muhan},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={365--376},
  year={2024}
}
```
