from __future__ import division
from __future__ import print_function

import json
import os.path
from abc import abstractmethod

# utils
import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.datasets import Planetoid, Amazon, CoraFull, Coauthor, HeterophilousGraphDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected

def load_dataset(args):
    """
    Create a dictionary to keep the mapping relationship between dataset and dataset_class
    key: "dataset" -> value: dataset class #
    for example, "Cora" -> 0
    class number to dataset_class (in PyG): 0: "Planetoid"; 1: "MyAmazon"; 2: "PygNodePropPredDataset"
    """
    relation_dic = {}
    available_datasets = [["Cora", "CiteSeer", "PubMed"],
                          ["Computers", "Photo"],
                          ["ogbn-arxiv", "ogbn-mag", "ogbn-products]"],
                          ["CoraFull"],
                          ["Coauthor-CS"],
                          ["Roman-empire"]]
    for cls, dataset_lst in enumerate(available_datasets):
        for dataset in dataset_lst:
            relation_dic[dataset] = cls

    # Load dataset from args.data_path folder
    # If no existing dataset, it will be automatically downloaded
    if relation_dic[args.dataset] == 0:
        dataset = Planetoid(args.data_path, args.dataset, transform=T.NormalizeFeatures())
    elif relation_dic[args.dataset] == 1:
        dataset = Amazon(args.data_path, args.dataset, transform=T.NormalizeFeatures())
    elif relation_dic[args.dataset] == 2:
        dataset = PygNodePropPredDataset(root=args.data_path, name=args.dataset, transform=T.NormalizeFeatures())
        # change arxiv to undirected graph
        edge_index = to_undirected(dataset._data.edge_index)
        set_dataset_attr(dataset, 'edge_index', edge_index,
                         edge_index.shape[1])
        # dataset.y = dataset.y.squeeze()
        # print(dataset[0].y)
    elif relation_dic[args.dataset] == 3:
        dataset = CoraFull(root=args.data_path+'/'+args.dataset, transform=T.NormalizeFeatures())
    elif relation_dic[args.dataset] == 4:
        dataset = Coauthor(root=args.data_path, name="CS", transform=T.NormalizeFeatures())
    elif relation_dic[args.dataset] == 5:
        dataset = HeterophilousGraphDataset(root=args.data_path, name="Roman-empire", transform=T.NormalizeFeatures())
        edge_index = to_undirected(dataset._data.edge_index)
        set_dataset_attr(dataset, 'edge_index', edge_index,
                         edge_index.shape[1])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Please choose from {available_datasets}")

    print(f"Successfully load dataset: {args.dataset} from {args.data_path}.")
    return dataset


# dataset = load_dataset(args)
# data = dataset[0]
# datamanager = FewShotDataManager(data, args)
# train_loader = datamanager.get_data_loader(0)
# val_loader = datamanager.get_data_loader(1)
# test_loader = datamanager.get_data_loader(2)


class DataManager:
    @abstractmethod
    def get_data_loader(self, mode):
        pass


class FewShotDataManager(DataManager):
    def __init__(self, data, args):
        super(FewShotDataManager, self).__init__()
        self.args = args
        data.y = data.y.squeeze()
        self.dataset = FewShotDataset(data, args, args.k_shot + args.q_query)
        self.split = self.dataset.split

    def get_data_loader(self, mode):
        # mode: 0->train, 1->val, 2->test
        class_list = self.dataset.__getclass__(mode)
        sampler = EpisodeBatchSampler(self.args, class_list, mode)
        # sampler = BatchSampler(EpisodeBatchSampler(self.args, class_list), batch_size=10, drop_last=False)
        # sampler = np.concatenate(list(sampler), axis=1)
        # print(sampler)
        data_loader_params = dict(batch_sampler=sampler,
                                  num_workers=self.args.num_workers,
                                  pin_memory=False)
        data_loader = DataLoader(self.dataset, **data_loader_params)
        return data_loader

    def get_dataset(self):
        return self.dataset


class FewShotDataset(Dataset):
    def __init__(self, data, args, batch_size):
        self.data = data
        self.args = args
        self.batch_size = batch_size

        self.cls_split_lst = self.class_split()
        self.cls_dataloader = self.create_subdataloader()
        self.split = self.get_split_index()

    def class_split(self):
        """
        Split class for train/val/test in meta learning setting.
        Save as list: [[train_class_index], [val_class_index], [test_class_index]]
        """
        cls_split_file = self.args.data_path + '/' + self.args.dataset + '_class_split.json'

        if os.path.isfile(cls_split_file) and False:
            # load list if exists
            with open(cls_split_file, 'rb') as f:
                cls_split_lst = json.load(f)
                print('Complete: Load class split info from %s .' % cls_split_file)

        else:
            # create list according to class_split_ratio and save
            label = torch.unique(self.data.y).cpu().detach()
            # if CoraFull dataset, ignore 68,69 label since they only have 15/29 samples
            if label.size(0) == 70:
                label = label[:-2]
            # randomly shuffle
            label = label.index_select(0, torch.randperm(label.shape[0]))
            train_class, val_class, test_class = torch.split(label, self.args.class_split_ratio)
            cls_split_lst = [train_class.tolist(), val_class.tolist(), test_class.tolist()]

            # with open(cls_split_file, 'w') as f:
            #     json.dump(cls_split_lst, f)
            #     print('Complete: Save class split info to %s .' % cls_split_file)
        print(cls_split_lst)
        return cls_split_lst

    def label_to_index(self) -> (dict, torch.tensor):
        """
        Generate a dictionary mapping labels to index list
        :return: dictionary: {label: [list of index]}
        """
        label = torch.unique(self.data.y)
        label2index = {}
        for i in label:
            label2index[int(i)] = torch.nonzero(self.data.y == i).squeeze()

        return label2index, label

    def create_subdataloader(self):
        """
        :return: list of subdataloaders for each class i
        """
        label2index, label = self.label_to_index()
        cls_dataloader = []
        cls_dataloader_params = dict(batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=self.args.num_workers,
                                     pin_memory=False)
        for c in label:
            cls_dataset = ClassDataset(label2index[int(c)])
            cls_dataloader.append(DataLoader(cls_dataset, **cls_dataloader_params))

        return cls_dataloader

    def get_split_index(self):
        """
        :return: dictionary that contains the node index for each split
        """
        label2index, label = self.label_to_index()
        cls_split_lst = self.cls_split_lst
        split = {
            'train': [],
            'valid': [],
            'test': []
        }

        for c in label:
            if c in cls_split_lst[0]:
                split['train'].extend([int(idx) for idx in label2index[int(c)]])
            elif c in cls_split_lst[1]:
                split['valid'].extend([int(idx) for idx in label2index[int(c)]])
            elif c in cls_split_lst[2]:
                split['test'].extend([int(idx) for idx in label2index[int(c)]])
            else:
                print("label %s does not belong to any class." % c)

        return split

    def __getitem__(self, class_index):
        return next(iter(self.cls_dataloader[class_index])), class_index

    def __len__(self):
        # mode = 0 -> train; 1 -> validation; 2 -> test
        return len(torch.unique(self.data.y))

    def __getclass__(self, mode):
        # return available classes under current mode (train/val/test)
        # print(self.cls_split_lst)
        return self.cls_split_lst[mode]


class EpisodeBatchSampler(object):
    def __init__(self, args, class_list, mode):
        # TODO: change value of episode to some variables
        self.episode = 1
        self.n_way = args.n_way
        self.class_list = class_list
        self.mode = mode
        self.test_num = args.task_num

    def __len__(self):
        return self.episode

    def __iter__(self):
        for i in range(self.episode):
            batch_class = []
            task_num = self.test_num if self.mode!=0 else 1
            for j in range(task_num):
                batch_class.append(np.random.choice(self.class_list, self.n_way, replace=False))
            yield np.concatenate(batch_class)


class ClassDataset(Dataset):
    def __init__(self, label_index):
        self.label_index = label_index

    def __getitem__(self, i):
        return self.label_index[i]

    def __len__(self):
        return self.label_index.shape[0]


class IndexDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = data.x.size(0)

    def __getitem__(self, i):
        return i

    def __len__(self):
        return self.len

class IndexDataset3(Dataset):
    def __init__(self, data, test_idx):
        self.data = data
        self.data_idx = torch.arange(data.x.size(0))

        test_idx_tensor = torch.tensor(test_idx)
        mask = torch.isin(self.data_idx, test_idx_tensor)
        mask_not_in_list = torch.logical_not(mask)
        self.train_val_idx = self.data_idx[mask_not_in_list]
        self.len = self.train_val_idx.size(0)



    def __getitem__(self, i):
        return self.train_val_idx[i]

    def __len__(self):
        return self.len

def set_dataset_attr(dataset, name, value, size):
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)
