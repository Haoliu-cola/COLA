
import lightning.pytorch as pl
from argparse import ArgumentParser

from typing import Dict, List
import torch.optim as optim
import numpy as np

import torch
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score, Recall, Precision, AUROC, ConfusionMatrix
import torch.nn.functional as F
import torch_geometric

from torch_geometric.utils import k_hop_subgraph

from model import MLP, GNNModel
from utils import precision_at_k, map_class

from sklearn.linear_model import LogisticRegression as SKLR
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

import ignite.distributed as idist
from utils import normalize_0to1

# GFS
class GFS(pl.LightningModule):
    def __init__(self,
                 args,
                 data,
                 augmentor,
                 test_idx,
                 encoder_momentum: float = 0.999,
                 encoder_depth=4,
                 head_depth=2,
                 softmax_temperature: float = 0.5,
                 learning_rate: float = 1e-3,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])
        self.args = args
        self.data = data
        self.test_idx = test_idx

        self.aug1, self.aug2 = augmentor

        # create encoders and projection heads
        self.encoder_q, self.encoder_k, self.pretraining_head_q, self.pretraining_head_k = self._init_encoders(args)
        # initialize weights
        self.encoder_q.apply(self._init_weights)
        self.pretraining_head_q.apply(self._init_weights)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            paramh_k.data.copy_(paramh_q.data)  # initialize
            paramh_k.requires_grad = False  # not update by gradient

        # # Add prediction head
        # self.prediction_head = MLP(args.out_dim, args.out_dim)
        # self.prediction_head.apply(self._init_weights)

        # create the queue
        self.register_buffer("queue", torch.randn(args.out_dim, args.num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def _init_encoders(self, args):
        if args.base_model == 'MLP':
            encoder_q = MLP(args.input_dim, args.out_dim, args.num_layers)
            encoder_k = MLP(args.input_dim, args.out_dim, args.num_layers)
        else:
            encoder_q = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )
            encoder_k = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )

        # Initialize pretraining_head with MLP
        pretraining_head_q = MLP(args.out_dim, args.out_dim)
        pretraining_head_k = MLP(args.out_dim, args.out_dim)

        return encoder_q, encoder_k, pretraining_head_q, pretraining_head_k

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            em = self.hparams.encoder_momentum
            paramh_k.data = paramh_k.data * em + paramh_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_ptr, queue):
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.args.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.args.num_negatives  # move pointer

        queue_ptr[0] = ptr


    def forward(self,
                data: torch_geometric.data.data,
                queue):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index, edge_weight)


        # GNN model
        z1 = self.encoder_q(x1, edge_index1, edge_weight1)
        z1 = self.pretraining_head_q(z1)
        # z1 = self.prediction_head(z1)
        z1 = nn.functional.normalize(z1, dim=1)

        with torch.no_grad():
            z2 = self.encoder_k(x2, edge_index2, edge_weight2)
            #z2 = self.pretraining_head_k(z2)
            z2 = nn.functional.normalize(z2, dim=1)

        return z1, z2

    def _calculate_fs_loss(self, z1, z2, query_idx, queue, queue_mask=None):
        z1_query = z1[query_idx]
        z2_query = z2[query_idx]

        sim = torch.einsum("nc,ck->nk", [z2_query, queue])
        if self.args.label_mask or self.args.khop_mask:
            sim *= queue_mask

        print('sim info:')

        print(torch.max(sim, dim=1).values)
        print(torch.min(sim, dim=1).values)

        topk_idx = torch.topk(sim, k=self.args.k_shot, dim=1, largest=True).indices
        support_embeddings = queue[:, topk_idx]
        prototypes = support_embeddings.mean(dim=2)

        loss_fs = (torch.mm(z1_query, support_embeddings.reshape(self.args.out_dim, -1)).div(
            self.args.temperature2).logsumexp(dim=1) - z1_query.mul(prototypes.T).div(self.args.temperature2).sum(
            dim=1)).mean()

        return loss_fs

    def _calculate_cl_loss(self, z1, z2, queue):
        # compute logits for moco
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [z1, z2]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [z1, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply_temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        loss_mc = F.cross_entropy(logits.float(), labels.long())

        return loss_mc


    def _calculate_mask(self, query_idx):
        data = self.data
        queue_mask = None
        mask = torch.ones([self.args.n_way, self.args.num_samples])

        # label mask: [n_way, len(queue)]
        if self.args.label_mask:
            query_label = data.y[query_idx].view(-1,1)
            mask = data.y.T == query_label
            if self.args.label_mask == 2:
                for i, idx  in enumerate(query_idx):
                    if int(idx) in self.test_idx:
                        mask[i, :] = 1

        elif self.args.khop_mask != 0:
            mask = torch.zeros([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                subset, _, _, _ = k_hop_subgraph(int(idx), self.args.khop_mask, data.edge_index)
                mask[row, subset] = 1

        elif self.args.self_mask:
            mask = torch.ones([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                mask[row, idx] = 0

        queue_mask = mask.repeat_interleave(int(self.args.num_negatives / mask.size()[1]), dim=-1)

        return queue_mask


    def training_step(self, batch, batch_idx):

        self._momentum_update_key_encoder()
        z1, z2 = self(data=self.data.to(batch.device), queue=self.queue)

        # calculate few-shot loss
        loss_fs = 0
        task_num = batch.size()[0] / self.args.n_way
        assert int(task_num) == self.args.train_task_num
        for i in range(int(task_num)):
            query_idx = batch[i*self.args.n_way:i*self.args.n_way+self.args.n_way]
            queue_mask = self._calculate_mask(query_idx).to(batch.device)
            loss_fs += self._calculate_fs_loss(z1, z2, query_idx=query_idx, queue=self.queue, queue_mask=queue_mask)
        loss_fs /= task_num

        # calculate moco_v2 loss
        loss_mc = self._calculate_cl_loss(z1, z2, queue=self.queue)

        loss_sum = self.args.fs_rate * loss_fs + (1 - self.args.fs_rate) * loss_mc

        log = {"train_loss": loss_sum, "train_loss_fs": loss_fs, "train_loss_mc": loss_mc}
        self.log_dict(log)

        self._dequeue_and_enqueue(z2, queue=self.queue, queue_ptr=self.queue_ptr)

        return loss_sum

    def fs_test(self, batch, data, args, mode="val"):

        task, target = batch

        encoder_model = self.encoder_q
        encoder_model.eval()
        embeddings = encoder_model(data.x,
                                   data.edge_index,
                                   data.edge_attr).detach().cpu().numpy()

        test_acc_all = []
        for i in range(args.task_num):
            task_idx = i * args.n_way
            random_support = torch.randperm(args.n_way * args.k_shot)
            random_query = torch.randperm(args.n_way * args.q_query)

            support_idx = task[task_idx:task_idx + args.n_way, :args.k_shot].reshape(1, -1).squeeze()[random_support].detach().cpu().numpy()
            query_idx = task[task_idx:task_idx + args.n_way, args.k_shot:].reshape(1, -1).squeeze()[random_query].detach().cpu().numpy()

            task_target = target[task_idx:task_idx + args.n_way]
            support_target = map_class(task_target, args.k_shot)[random_support]
            query_target = map_class(task_target, args.q_query)[random_query]

            emb_train = embeddings[support_idx]
            emb_test = embeddings[query_idx]

            if args.classifier == 'LR':
                clf = SKLR(solver='lbfgs', max_iter=1000, multi_class='auto').fit(emb_train,
                                                                              support_target.detach().numpy())
            elif args.classifier == 'SVC':
                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                clf = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0).fit(emb_train,
                                                                                                 support_target.detach().numpy())

            test_acc = clf.score(emb_test, query_target.detach().numpy())
            test_acc_all.append(test_acc)

        final_mean = np.mean(test_acc_all)
        final_std = np.std(test_acc_all)
        final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))

        log = {mode+"_acc": final_mean, mode+"_std": final_std, mode+"_interval": final_interval}
        self.log_dict(log,
                      prog_bar=True,
                      batch_size=args.task_num,
                      add_dataloader_idx=False
                      )

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="val")
        elif dataloader_idx == 1:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="t_val")

    # def validation_step(self, batch, batch_idx):
    #     self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="val")

    def test_step(self, batch, batch_idx):
        self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.max_epochs, eta_min=self.args.lr / 100
        )
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     self.args.lr,
        #     momentum=self.hparams.momentum,
        #     weight_decay=self.args.weight_decay,
        # )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     self.trainer.max_epochs,
        # )
        return [optimizer], [lr_scheduler]

# GFS2
class GFS2(pl.LightningModule):
    def __init__(self,
                 args,
                 data,
                 augmentor,
                 encoder_momentum: float = 0.999,
                 encoder_depth=4,
                 head_depth=2,
                 softmax_temperature: float = 0.5,
                 learning_rate: float = 1e-3,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])
        self.args = args
        self.data = data

        self.aug1, self.aug2, self.aug3 = augmentor



        # create encoders and projection heads
        self.encoder_q, self.encoder_k, self.pretraining_head_q, self.pretraining_head_k = self._init_encoders(args)
        # initialize weights
        self.encoder_q.apply(self._init_weights)
        self.pretraining_head_q.apply(self._init_weights)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            paramh_k.data.copy_(paramh_q.data)  # initialize
            paramh_k.requires_grad = False  # not update by gradient

        # # Add prediction head
        # self.prediction_head = MLP(args.out_dim, args.out_dim)
        # self.prediction_head.apply(self._init_weights)

    def _init_encoders(self, args):
        if args.base_model == 'MLP':
            encoder_q = MLP(args.input_dim, args.out_dim, args.num_layers)
            encoder_k = MLP(args.input_dim, args.out_dim, args.num_layers)
        else:
            encoder_q = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )
            encoder_k = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )

        # Initialize pretraining_head with MLP
        pretraining_head_q = MLP(args.out_dim, args.out_dim)
        pretraining_head_k = MLP(args.out_dim, args.out_dim)

        return encoder_q, encoder_k, pretraining_head_q, pretraining_head_k

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            em = self.hparams.encoder_momentum
            paramh_k.data = paramh_k.data * em + paramh_q.data * (1.0 - em)

    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, keys, queue_ptr, queue):
    #     batch_size = keys.shape[0]
    #
    #     ptr = int(queue_ptr)
    #     assert self.args.num_negatives % batch_size == 0  # for simplicity
    #
    #     # replace the keys at ptr (dequeue and enqueue)
    #     queue[:, ptr: ptr + batch_size] = keys.T
    #     ptr = (ptr + batch_size) % self.args.num_negatives  # move pointer
    #
    #     queue_ptr[0] = ptr


    def forward(self,
                data: torch_geometric.data.data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index, edge_weight)
        x3, edge_index3, edge_weight3 = self.aug3(x, edge_index, edge_weight)


        # GNN model
        z1 = self.encoder_q(x1, edge_index1, edge_weight1)
        z1 = self.pretraining_head_q(z1)
        # z1 = self.prediction_head(z1)
        z1 = nn.functional.normalize(z1, dim=1)

        with torch.no_grad():
            # TODO: Test use momentum or non-momentum encoder for Lookup embeddings
            z2 = self.encoder_k(x2, edge_index2, edge_weight2)
            #z2 = self.pretraining_head_k(z2)
            z2 = nn.functional.normalize(z2, dim=1)

            z3 = self.encoder_q(x3, edge_index3, edge_weight3)
            # z3 = self.pretraining_head_k(z3)
            z3 = nn.functional.normalize(z3, dim=1)


        return z1, z2, z3

    def _calculate_fs_loss(self, z1, z2, z3, query_idx, queue_mask=None):
        z1_query = z1[query_idx]
        z2_query = z2[query_idx]
        #z3_query = z3[query_idx]

        sim = torch.einsum("nc,bc->nb", [z2_query, z3])
        #sim = distributed_sinkhorn(sim, normalization="row")
        if self.args.label_mask or self.args.khop_mask:
            sim *= queue_mask

        topk_idx = torch.topk(sim, k=self.args.k_shot, dim=1, largest=True).indices
        topk_idx = torch.cat((topk_idx, query_idx.view(-1,1)), 1)
        #assert topk_idx.size()[1] == self.args.k_shot + 1
        support_embeddings = z3[topk_idx]
        #print(support_embeddings.size())
        prototypes = support_embeddings.mean(dim=1)

        # change negative sample to all the samples
        # support_embeddings = z3
        # print(support_embeddings.size())

        loss_fs = (torch.mm(z1_query, support_embeddings.reshape(self.args.out_dim, -1)).div(
            self.args.temperature2).logsumexp(dim=1) - z1_query.mul(prototypes).div(self.args.temperature2).sum(
            dim=1)).mean()

        return loss_fs


    def _calculate_mask(self, query_idx):
        data = self.data
        queue_mask = None
        mask = torch.ones([self.args.n_way, self.args.num_samples])

        # label mask: [n_way, len(queue)]
        if self.args.label_mask:
            print('use label mask')
            query_label = data.y[query_idx].view(-1,1)
            mask = data.y.T == query_label

        elif self.args.khop_mask != 0:
            mask = torch.zeros([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                subset, _, _, _ = k_hop_subgraph(int(idx), self.args.khop_mask, data.edge_index)
                mask[row, subset] = 1

        elif self.args.self_mask:
            mask = torch.ones([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                mask[row, idx] = 0

        return mask


    def training_step(self, batch, batch_idx):
        print('GFS2 begin!')
        self._momentum_update_key_encoder()
        z1, z2, z3 = self(data=self.data.to(batch.device))
        assert z1.requires_grad == True
        assert z2.requires_grad == False
        assert z3.requires_grad == False

        # calculate few-shot loss
        loss_fs1 = 0
        loss_fs2 = 0
        task_num = batch.size()[0] / self.args.n_way
        assert int(task_num) == self.args.train_task_num
        for i in range(int(task_num)):
            query_idx = batch[i*self.args.n_way:i*self.args.n_way+self.args.n_way]
            queue_mask = self._calculate_mask(query_idx).to(batch.device)
            if self.args.compare_mode == 'three':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                #loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'z1toz23':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'z23toz1':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z1, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z1, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'm1m2':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode =='m4':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z3, query_idx=query_idx, queue_mask=queue_mask)

        loss_fs1 /= task_num
        loss_fs2 /= task_num
        loss_fs = (loss_fs1 + loss_fs2) / 2

        log = {"train_loss_fs": loss_fs, "train_loss_fs1": loss_fs1, "train_loss_fs2": loss_fs2}
        self.log_dict(log)
        #print(type(loss_fs))

        return loss_fs

    def fs_test(self, batch, data, args, mode="val"):

        task, target = batch

        encoder_model = self.encoder_q
        encoder_model.eval()
        embeddings = encoder_model(data.x,
                                   data.edge_index,
                                   data.edge_attr).detach().cpu().numpy()

        test_acc_all = []
        for i in range(args.task_num):
            task_idx = i * args.n_way
            random_support = torch.randperm(args.n_way * args.k_shot)
            random_query = torch.randperm(args.n_way * args.q_query)

            support_idx = task[task_idx:task_idx + args.n_way, :args.k_shot].reshape(1, -1).squeeze()[random_support].detach().cpu().numpy()
            query_idx = task[task_idx:task_idx + args.n_way, args.k_shot:].reshape(1, -1).squeeze()[random_query].detach().cpu().numpy()

            task_target = target[task_idx:task_idx + args.n_way]
            support_target = map_class(task_target, args.k_shot)[random_support]
            query_target = map_class(task_target, args.q_query)[random_query]

            emb_train = embeddings[support_idx]
            emb_test = embeddings[query_idx]

            if args.classifier == 'LR':
                clf = SKLR(solver='lbfgs', max_iter=1000, multi_class='auto').fit(emb_train,
                                                                              support_target.detach().numpy())
            elif args.classifier == 'SVC':
                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                clf = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0).fit(emb_train,
                                                                                                 support_target.detach().numpy())

            test_acc = clf.score(emb_test, query_target.detach().numpy())
            test_acc_all.append(test_acc)

        final_mean = np.mean(test_acc_all)
        final_std = np.std(test_acc_all)
        final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))

        log = {mode+"_acc": final_mean, mode+"_std": final_std, mode+"_interval": final_interval}
        self.log_dict(log,
                      prog_bar=True,
                      batch_size=args.task_num,
                      add_dataloader_idx=False
                      )

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="val")
        elif dataloader_idx == 1:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="t_val")


    def test_step(self, batch, batch_idx):
        self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.max_epochs
        )
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     self.args.lr,
        #     momentum=self.hparams.momentum,
        #     weight_decay=self.args.weight_decay,
        # )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     self.trainer.max_epochs,
        # )
        return [optimizer], [lr_scheduler]


# GFS3
class GFS3(pl.LightningModule):
    def __init__(self,
                 args,
                 data,
                 augmentor,
                 test_idx,
                 encoder_momentum: float = 0.999,
                 encoder_depth=4,
                 head_depth=2,
                 softmax_temperature: float = 0.5,
                 learning_rate: float = 1e-3,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])
        self.args = args
        self.data = data
        self.test_idx = test_idx

        self.aug1, self.aug2, self.aug3 = augmentor
        self.training_step_outputs = []

        # create encoders and projection heads
        self.encoder_q, self.encoder_k, self.pretraining_head_q, self.pretraining_head_k = self._init_encoders(args)
        # initialize weights
        self.encoder_q.apply(self._init_weights)
        self.pretraining_head_q.apply(self._init_weights)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            paramh_k.data.copy_(paramh_q.data)  # initialize
            paramh_k.requires_grad = False  # not update by gradient


    def _init_encoders(self, args):
        if args.base_model == 'MLP':
            encoder_q = MLP(args.input_dim, args.out_dim, args.num_layers)
            encoder_k = MLP(args.input_dim, args.out_dim, args.num_layers)
        else:
            encoder_q = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )
            encoder_k = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )

        # Initialize pretraining_head with MLP
        pretraining_head_q = MLP(args.out_dim, args.out_dim)
        pretraining_head_k = MLP(args.out_dim, args.out_dim)

        return encoder_q, encoder_k, pretraining_head_q, pretraining_head_k

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            if self.current_epoch > 10:
                if self.args.em_scd == 1:
                    # Schedule em
                    max_em = 0.999
                    em += (self.current_epoch-10) / 100
                    em = min(em, max_em)
                    self.log('momentum',em)
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            em = self.hparams.encoder_momentum
            paramh_k.data = paramh_k.data * em + paramh_q.data * (1.0 - em)


    def forward(self,
                data: torch_geometric.data.data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index, edge_weight)
        x3, edge_index3, edge_weight3 = self.aug3(x, edge_index, edge_weight)


        # GNN model
        z1 = self.encoder_q(x1, edge_index1, edge_weight1)
        z1 = self.pretraining_head_q(z1)
        #z1 = self.prediction_head(z1)
        z1 = nn.functional.normalize(z1, dim=1)

        with torch.no_grad():
            z2 = self.encoder_k(x2, edge_index2, edge_weight2)
            #z2 = self.pretraining_head_q(z2)
            z2 = nn.functional.normalize(z2, dim=1)

            z3 = self.encoder_k(x3, edge_index3, edge_weight3)
            #z3 = self.pretraining_head_q(z3)
            z3 = nn.functional.normalize(z3, dim=1)


        return z1, z2, z3

    def _calculate_fs_loss(self, z1, z2, z3, query_idx, queue_mask=None):
        z1_query = z1[query_idx]
        z2_query = z2[query_idx]
        #z3_query = z3[query_idx]

        sim = torch.einsum("nc,bc->nb", [z2_query, z3])
        #print(sim)
        if self.args.label_mask or self.args.khop_mask:
            sim *= queue_mask
        #print('sim info:')

        # print(torch.max(sim,dim=1).values)
        # print(torch.min(sim,dim=1).values)

        # randomly select k samples
        # randk_idx = torch.zeros([self.args.n_way, self.args.k_shot], dtype=torch.long)
        # for i in range(self.args.n_way):
        #     true_indices = torch.nonzero(queue_mask[i], as_tuple=True)[0]
        #     shuffled_indices = torch.randperm(true_indices.size(0))
        #     randk_idx[i] = true_indices[shuffled_indices[:self.args.k_shot]]
        # support_embeddings = z3[randk_idx]

        # select according to topk similarity
        k = self.args.k_shot * self.args.k_rate
        topnk_idx = torch.topk(sim, k=k, dim=1, largest=True).indices
        topk_idx = topnk_idx[:, torch.randperm(topnk_idx.size(1))[:self.args.k_shot]]
        #topk_idx = torch.cat((topk_idx, query_idx.view(-1, 1)), 1)

        true_label = 0
        total_match = self.args.k_shot * self.args.n_way
        #total_match = self.args.n_way * 1
        for i in range(self.args.n_way):
            true_label +=  (self.data.y[query_idx[i]] == self.data.y[topk_idx[i]]).sum()
        true_ratio = true_label / total_match
        #print(true_ratio)

        # n_way x k_shot x out_dim
        support_embeddings = z3[topk_idx]

        prototypes = support_embeddings.mean(dim=1)

        support_embeddings = support_embeddings.reshape(-1, self.args.out_dim).transpose(0, 1)

        # TAITONGLA
        loss_fs = torch.mm(z1_query, support_embeddings).div(self.args.temperature2).logsumexp(dim=1) - z1_query.mul(prototypes).div(self.args.temperature2).sum(dim=1)

        # negative samples: all
        # loss_fs = torch.mm(z1_query, z3.T).div(self.args.temperature2).logsumexp(
        #     dim=1) - z1_query.mul(prototypes).div(self.args.temperature2).sum(dim=1)

        # loss_fs = torch.mm(z1_query, z3[torch.randperm(z3.size(0))[:512]].T).div(self.args.temperature2).logsumexp(dim=1) - z1_query.mul(
        #     prototypes).div(self.args.temperature2).sum(dim=1)
        # (torch.randperm(z3.size(0))[:32])

        # print(torch.mm(z1_query,support_embeddings))
        # print(loss_fs)
        # losssim = torch.mm(z1_query,z3.transpose(0,1))
        # print('losssim info:')
        # print(losssim)
        # print(torch.max(losssim, dim=1).values)
        # print(torch.min(losssim, dim=1).values)

        loss_fs = loss_fs.mean()

        #loss_fs += 2

        return loss_fs, true_ratio

    def _calculate_label_penalty(self, z1, z2, query_idx):
        z1_query = z1[query_idx]
        sim = torch.einsum("nc,bc->nb", [z1_query, z2])
        sim = normalize_0to1(sim)

        data = self.data
        query_label = data.y[query_idx].view(-1, 1)
        mask = data.y.T == query_label

        loss_penalty = 0
        count = 0
        loss_function = nn.BCELoss()

        for i, idx in enumerate(query_idx):
            if int(idx) in self.test_idx:
                continue
            else:
                #loss_penalty += (sim[i,:] - mask[i, :].to(torch.int32)).norm(dim=0, p=2)
                loss_penalty += loss_function(sim[i, :], mask[i, :].to(torch.float32))
                count += 1

        if count > 0:
            return loss_penalty/count
        else:
            return loss_penalty


    def _calculate_mask(self, query_idx):
        data = self.data
        queue_mask = None
        mask = torch.ones([self.args.n_way, self.args.num_samples])

        # label mask: [n_way, len(queue)]
        if self.args.label_mask:
            query_label = data.y[query_idx].view(-1,1)
            mask = data.y.T == query_label
            neg_mask = data.y.T != query_label
            if self.args.label_mask == 2:
                for i, idx  in enumerate(query_idx):
                    if int(idx) in self.test_idx:
                        mask[i, :] = 1

        elif self.args.khop_mask != 0:
            mask = torch.zeros([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                subset, _, _, _ = k_hop_subgraph(int(idx), self.args.khop_mask, data.edge_index)
                mask[row, subset] = 1

        elif self.args.self_mask:
            mask = torch.ones([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                mask[row, idx] = 0

        return mask


    def training_step(self, batch, batch_idx):
        #print('GFS3 begins!')
        self._momentum_update_key_encoder()
        z1, z2, z3 = self(data=self.data.to(batch.device))
        assert z1.requires_grad == True
        assert z2.requires_grad == False
        assert z3.requires_grad == False

        # calculate few-shot loss
        loss_fs1 = 0
        loss_fs2 = 0
        true_ratio = 0
        task_num = batch.size()[0] / self.args.n_way
        assert int(task_num) == self.args.train_task_num
        for i in range(int(task_num)):
            query_idx = batch[i*self.args.n_way:i*self.args.n_way+self.args.n_way]
            queue_mask = self._calculate_mask(query_idx).to(batch.device)
            if self.args.compare_mode == 'm1':
                loss_fs11, true_ratio1 = self._calculate_fs_loss(z1=z1, z2=z2, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs22, true_ratio2 = self._calculate_fs_loss(z1=z1, z2=z3, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs1 += loss_fs11
                loss_fs2 += loss_fs22
                true_ratio += (true_ratio1 + true_ratio2) / 2
            elif self.args.compare_mode == 'm2':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'm3':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z1.detach(), query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z1.detach(), query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'm4':
                loss_fs11, true_ratio1 = self._calculate_fs_loss(z1=z1, z2=z1.detach(), z3=z1.detach(), query_idx=query_idx, queue_mask=queue_mask)
                loss_fs22, true_ratio2 = self._calculate_fs_loss(z1=z1, z2=z1.detach(), z3=z1.detach(), query_idx=query_idx, queue_mask=queue_mask)
                loss_fs1 += loss_fs11
                loss_fs2 += loss_fs22
                true_ratio += (true_ratio1 + true_ratio2) / 2
            elif self.args.compare_mode == 'm1m2':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z2, query_idx=query_idx, queue_mask=queue_mask)

        loss_fs1 /= task_num
        loss_fs2 /= task_num
        loss_fs = (loss_fs1 + loss_fs2) / 2
        true_ratio /= task_num
        self.log("true_ratio",true_ratio)

        # loss_penalty = self._calculate_label_penalty(z1=z1, z2=z2, query_idx=query_idx)
        # loss_penalty += self._calculate_label_penalty(z1=z2, z2=z3, query_idx=query_idx)
        # loss_fs += 0.5 * loss_penalty

        log = {"train_loss_fs": loss_fs, "train_loss_fs1": loss_fs1, "train_loss_fs2": loss_fs2}
        self.log_dict(log)
        #print(type(loss_fs))
        self.training_step_outputs.append(true_ratio)

        return loss_fs

    def on_train_epoch_end(self) -> None:

        epoch_average_true_ratio = torch.stack(self.training_step_outputs).mean()
        # epoch_average_loss = torch.stack(x['loss'] for x in self.training_step_outputs).mean()
        logs = {'true_ratio_epoch': epoch_average_true_ratio, 'step': self.current_epoch}
        self.log_dict(logs)
        self.training_step_outputs.clear()


    def fs_test(self, batch, data, args, mode="val"):

        task, target = batch

        encoder_model = self.encoder_q
        encoder_model.eval()
        embeddings = encoder_model(data.x,
                                   data.edge_index,
                                   data.edge_attr).detach().cpu().numpy()

        test_acc_all = []
        for i in range(args.task_num):
            task_idx = i * args.n_way
            random_support = torch.randperm(args.n_way * args.k_shot)
            random_query = torch.randperm(args.n_way * args.q_query)

            support_idx = task[task_idx:task_idx + args.n_way, :args.k_shot].reshape(1, -1).squeeze()[random_support].detach().cpu().numpy()
            query_idx = task[task_idx:task_idx + args.n_way, args.k_shot:].reshape(1, -1).squeeze()[random_query].detach().cpu().numpy()

            task_target = target[task_idx:task_idx + args.n_way]
            support_target = map_class(task_target, args.k_shot)[random_support]
            query_target = map_class(task_target, args.q_query)[random_query]

            emb_train = embeddings[support_idx]
            emb_test = embeddings[query_idx]

            if args.classifier == 'LR':
                clf = SKLR(solver='lbfgs', max_iter=1000, multi_class='auto').fit(emb_train,
                                                                              support_target.detach().numpy())
            elif args.classifier == 'SVC':
                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                clf = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0).fit(emb_train,
                                                                                                 support_target.detach().numpy())

            test_acc = clf.score(emb_test, query_target.detach().numpy())
            test_acc_all.append(test_acc)

        final_mean = np.mean(test_acc_all)
        final_std = np.std(test_acc_all)
        final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))

        log = {mode+"_acc": final_mean, mode+"_std": final_std, mode+"_interval": final_interval}
        self.log_dict(log,
                      prog_bar=True,
                      batch_size=args.task_num,
                      add_dataloader_idx=False
                      )

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="val")
        elif dataloader_idx == 1:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="t_val")


    def test_step(self, batch, batch_idx):
        self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.max_epochs
        )
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     self.args.lr,
        #     momentum=self.hparams.momentum,
        #     weight_decay=self.args.weight_decay,
        # )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     self.trainer.max_epochs,
        # )
        return [optimizer], [lr_scheduler]


# No momentum
class GFS4(pl.LightningModule):
    def __init__(self,
                 args,
                 data,
                 augmentor,
                 encoder_momentum: float = 0.999,
                 encoder_depth=4,
                 head_depth=2,
                 softmax_temperature: float = 0.5,
                 learning_rate: float = 1e-3,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])
        self.args = args
        self.data = data

        self.aug1, self.aug2, self.aug3 = augmentor

        # create encoders and projection heads
        self.encoder_q, self.encoder_k, self.pretraining_head_q, self.pretraining_head_k = self._init_encoders(args)
        # initialize weights
        self.encoder_q.apply(self._init_weights)
        self.pretraining_head_q.apply(self._init_weights)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            paramh_k.data.copy_(paramh_q.data)  # initialize
            paramh_k.requires_grad = False  # not update by gradient

        # # Add prediction head
        # self.prediction_head = MLP(args.out_dim, args.out_dim)
        # self.prediction_head.apply(self._init_weights)

        # # create the queue
        # self.register_buffer("queue", torch.randn(args.out_dim, args.num_negatives))
        # self.queue = nn.functional.normalize(self.queue, dim=0)
        #
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def _init_encoders(self, args):
        if args.base_model == 'MLP':
            encoder_q = MLP(args.input_dim, args.out_dim, args.num_layers)
            encoder_k = MLP(args.input_dim, args.out_dim, args.num_layers)
        else:
            encoder_q = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )
            encoder_k = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )

        # Initialize pretraining_head with MLP
        pretraining_head_q = MLP(args.out_dim, args.out_dim)
        pretraining_head_k = MLP(args.out_dim, args.out_dim)

        return encoder_q, encoder_k, pretraining_head_q, pretraining_head_k

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            em = self.hparams.encoder_momentum
            paramh_k.data = paramh_k.data * em + paramh_q.data * (1.0 - em)


    def forward(self,
                data: torch_geometric.data.data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index, edge_weight)
        x3, edge_index3, edge_weight3 = self.aug3(x, edge_index, edge_weight)


        # GNN model
        z1 = self.encoder_q(x1, edge_index1, edge_weight1)
        z1 = self.pretraining_head_q(z1)
        # z1 = self.prediction_head(z1)
        z1 = nn.functional.normalize(z1, dim=1)


        z2 = self.encoder_q(x2, edge_index2, edge_weight2)
        #z2 = self.pretraining_head_k(z2)
        z2 = nn.functional.normalize(z2, dim=1)

        z3 = self.encoder_q(x3, edge_index3, edge_weight3)
        # z3 = self.pretraining_head_k(z3)
        z3 = nn.functional.normalize(z3, dim=1)


        return z1, z2, z3

    def _calculate_fs_loss(self, z1, z2, z3, query_idx, queue_mask=None):
        z1_query = z1[query_idx]
        z2_query = z2[query_idx]
        #z3_query = z3[query_idx]

        sim = torch.einsum("nc,bc->nb", [z2_query, z3])
        if self.args.label_mask or self.args.khop_mask:
            sim *= queue_mask

        topk_idx = torch.topk(sim, k=self.args.k_shot, dim=1, largest=True).indices
        support_embeddings = z3[topk_idx]
        prototypes = support_embeddings.mean(dim=1)

        loss_fs = (torch.mm(z1_query, support_embeddings.reshape(self.args.out_dim, -1)).div(
            self.args.temperature2).logsumexp(dim=1) - z1_query.mul(prototypes).div(self.args.temperature2).sum(
            dim=1)).mean()

        return loss_fs



    def _calculate_mask(self, query_idx):
        data = self.data
        queue_mask = None
        mask = torch.ones([self.args.n_way, self.args.num_samples])

        # label mask: [n_way, len(queue)]
        if self.args.label_mask:
            query_label = data.y[query_idx].view(-1,1)
            mask = data.y.T == query_label

        elif self.args.khop_mask != 0:
            mask = torch.zeros([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                subset, _, _, _ = k_hop_subgraph(int(idx), self.args.khop_mask, data.edge_index)
                mask[row, subset] = 1

        elif self.args.self_mask:
            mask = torch.ones([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                mask[row, idx] = 0

        return mask


    def training_step(self, batch, batch_idx):

        self._momentum_update_key_encoder()
        z1, z2, z3 = self(data=self.data.to(batch.device))
        assert z1.requires_grad == True
        assert z2.requires_grad == True
        assert z3.requires_grad == True


        # calculate few-shot loss
        loss_fs1 = 0
        loss_fs2 = 0
        task_num = batch.size()[0] / self.args.n_way
        assert int(task_num) == self.args.train_task_num
        for i in range(int(task_num)):
            query_idx = batch[i*self.args.n_way:i*self.args.n_way+self.args.n_way]
            queue_mask = self._calculate_mask(query_idx).to(batch.device)
            if self.args.compare_mode == 'three':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'z1toz23':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'z23toz1':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z1, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z1, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'm1m2':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z2, query_idx=query_idx, queue_mask=queue_mask)

        loss_fs1 /= task_num
        loss_fs2 /= task_num
        loss_fs = (loss_fs1 + loss_fs2) / 2

        log = {"train_loss_fs": loss_fs, "train_loss_fs1": loss_fs1, "train_loss_fs2": loss_fs2}
        self.log_dict(log)
        print(type(loss_fs))

        return loss_fs

    def fs_test(self, batch, data, args, mode="val"):

        task, target = batch

        encoder_model = self.encoder_q
        encoder_model.eval()
        embeddings = encoder_model(data.x,
                                   data.edge_index,
                                   data.edge_attr).detach().cpu().numpy()

        test_acc_all = []
        for i in range(args.task_num):
            task_idx = i * args.n_way
            random_support = torch.randperm(args.n_way * args.k_shot)
            random_query = torch.randperm(args.n_way * args.q_query)

            support_idx = task[task_idx:task_idx + args.n_way, :args.k_shot].reshape(1, -1).squeeze()[random_support].detach().cpu().numpy()
            query_idx = task[task_idx:task_idx + args.n_way, args.k_shot:].reshape(1, -1).squeeze()[random_query].detach().cpu().numpy()

            task_target = target[task_idx:task_idx + args.n_way]
            support_target = map_class(task_target, args.k_shot)[random_support]
            query_target = map_class(task_target, args.q_query)[random_query]

            emb_train = embeddings[support_idx]
            emb_test = embeddings[query_idx]

            if args.classifier == 'LR':
                clf = SKLR(solver='lbfgs', max_iter=1000, multi_class='auto').fit(emb_train,
                                                                              support_target.detach().numpy())
            elif args.classifier == 'SVC':
                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                clf = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0).fit(emb_train,
                                                                                                 support_target.detach().numpy())

            test_acc = clf.score(emb_test, query_target.detach().numpy())
            test_acc_all.append(test_acc)

        final_mean = np.mean(test_acc_all)
        final_std = np.std(test_acc_all)
        final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))

        log = {mode+"_acc": final_mean, mode+"_std": final_std, mode+"_interval": final_interval}
        self.log_dict(log,
                      prog_bar=True,
                      batch_size=args.task_num,
                      add_dataloader_idx=False
                      )

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="val")
        elif dataloader_idx == 1:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="t_val")


    def test_step(self, batch, batch_idx):
        self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.max_epochs
        )
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     self.args.lr,
        #     momentum=self.hparams.momentum,
        #     weight_decay=self.args.weight_decay,
        # )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     self.trainer.max_epochs,
        # )
        return [optimizer], [lr_scheduler]

class GFS5(pl.LightningModule):
    def __init__(self,
                 args,
                 data,
                 augmentor,
                 encoder_momentum: float = 0.999,
                 encoder_depth=4,
                 head_depth=2,
                 softmax_temperature: float = 0.5,
                 learning_rate: float = 1e-3,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])
        self.args = args
        self.data = data

        self.aug1, self.aug2, self.aug3 = augmentor

        # create encoders and projection heads
        self.encoder_q, self.encoder_k, self.pretraining_head_q, self.pretraining_head_k = self._init_encoders(args)
        # initialize weights
        self.encoder_q.apply(self._init_weights)
        self.pretraining_head_q.apply(self._init_weights)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            paramh_k.data.copy_(paramh_q.data)  # initialize
            paramh_k.requires_grad = False  # not update by gradient

    def _init_encoders(self, args):
        if args.base_model == 'MLP':
            encoder_q = MLP(args.input_dim, args.out_dim, args.num_layers)
            encoder_k = MLP(args.input_dim, args.out_dim, args.num_layers)
        else:
            encoder_q = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )
            encoder_k = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )

        # Initialize pretraining_head with MLP
        pretraining_head_q = MLP(args.out_dim, args.out_dim)
        pretraining_head_k = MLP(args.out_dim, args.out_dim)

        return encoder_q, encoder_k, pretraining_head_q, pretraining_head_k

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            em = self.hparams.encoder_momentum
            paramh_k.data = paramh_k.data * em + paramh_q.data * (1.0 - em)


    def forward(self,
                data: torch_geometric.data.data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index, edge_weight)
        x3, edge_index3, edge_weight3 = self.aug3(x, edge_index, edge_weight)


        # GNN model
        z1 = self.encoder_q(x1, edge_index1, edge_weight1)
        z1 = self.pretraining_head_q(z1)
        # z1 = self.prediction_head(z1)
        z1 = nn.functional.normalize(z1, dim=1)

        with torch.no_grad():
            # TODO: Test use momentum or non-momentum encoder for Lookup embeddings
            z2 = self.encoder_q(x2, edge_index2, edge_weight2)
            #z2 = self.pretraining_head_k(z2)
            z2 = nn.functional.normalize(z2, dim=1)

            z3 = self.encoder_k(x3, edge_index3, edge_weight3)
            # z3 = self.pretraining_head_k(z3)
            z3 = nn.functional.normalize(z3, dim=1)


        return z1, z2, z3

    def _calculate_fs_loss(self, z1, z2, z3, query_idx, queue_mask=None):
        z1_query = z1[query_idx]
        z2_query = z2[query_idx]
        z3_query = z3[query_idx]

        sim = torch.einsum("nc,bc->nb", [z3_query, z3])
        #sim = distributed_sinkhorn(sim, normalization="row")
        if self.args.label_mask or self.args.khop_mask:
            sim *= queue_mask

        topk_idx = torch.topk(sim, k=self.args.k_shot, dim=1, largest=True).indices
        #topk_idx = torch.cat((topk_idx, query_idx.view(-1,1)), 1)
        #assert topk_idx.size()[1] == self.args.k_shot + 1
        support_embeddings = z2[topk_idx]
        #print(support_embeddings.size())

        # compute prototypes: [n_way, out_dim]
        prototypes = support_embeddings.mean(dim=1)

        # change negative sample to all the samples
        # support_embeddings = z3
        # print(support_embeddings.size())

        loss_fs = (torch.mm(z1_query, support_embeddings.reshape(self.args.out_dim, -1)).div(
            self.args.temperature2).logsumexp(dim=1) - z1_query.mul(prototypes).div(self.args.temperature2).sum(
            dim=1)).mean()

        return loss_fs


    def _calculate_mask(self, query_idx):
        data = self.data
        queue_mask = None
        mask = torch.ones([self.args.n_way, self.args.num_samples])

        # label mask: [n_way, len(queue)]
        if self.args.label_mask:
            print('use label mask')
            query_label = data.y[query_idx].view(-1,1)
            mask = data.y.T == query_label

        elif self.args.khop_mask != 0:
            mask = torch.zeros([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                subset, _, _, _ = k_hop_subgraph(int(idx), self.args.khop_mask, data.edge_index)
                mask[row, subset] = 1

        elif self.args.self_mask:
            mask = torch.ones([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                mask[row, idx] = 0

        return mask


    def training_step(self, batch, batch_idx):

        self._momentum_update_key_encoder()
        z1, z2, z3 = self(data=self.data.to(batch.device))
        assert z1.requires_grad == True
        assert z2.requires_grad == False
        assert z3.requires_grad == False

        # calculate few-shot loss
        loss_fs1 = 0
        loss_fs2 = 0
        task_num = batch.size()[0] / self.args.n_way
        assert int(task_num) == self.args.train_task_num
        for i in range(int(task_num)):
            query_idx = batch[i*self.args.n_way:i*self.args.n_way+self.args.n_way]
            queue_mask = self._calculate_mask(query_idx).to(batch.device)
            if self.args.compare_mode == 'three':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                #loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'z1toz23':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'z23toz1':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z1, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z1, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'm1m2':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode =='m4':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z3, query_idx=query_idx, queue_mask=queue_mask)


        loss_fs1 /= task_num
        loss_fs2 /= task_num
        loss_fs = (loss_fs1 + loss_fs2) / 2

        log = {"train_loss_fs": loss_fs, "train_loss_fs1": loss_fs1, "train_loss_fs2": loss_fs2}
        self.log_dict(log)
        #print(type(loss_fs))

        return loss_fs

    def fs_test(self, batch, data, args, mode="val"):

        task, target = batch

        encoder_model = self.encoder_q
        encoder_model.eval()
        embeddings = encoder_model(data.x,
                                   data.edge_index,
                                   data.edge_attr).detach().cpu().numpy()

        test_acc_all = []
        for i in range(args.task_num):
            task_idx = i * args.n_way
            random_support = torch.randperm(args.n_way * args.k_shot)
            random_query = torch.randperm(args.n_way * args.q_query)

            support_idx = task[task_idx:task_idx + args.n_way, :args.k_shot].reshape(1, -1).squeeze()[random_support].detach().cpu().numpy()
            query_idx = task[task_idx:task_idx + args.n_way, args.k_shot:].reshape(1, -1).squeeze()[random_query].detach().cpu().numpy()

            task_target = target[task_idx:task_idx + args.n_way]
            support_target = map_class(task_target, args.k_shot)[random_support]
            query_target = map_class(task_target, args.q_query)[random_query]

            emb_train = embeddings[support_idx]
            emb_test = embeddings[query_idx]

            if args.classifier == 'LR':
                clf = SKLR(solver='lbfgs', max_iter=1000, multi_class='auto').fit(emb_train,
                                                                              support_target.detach().numpy())
            elif args.classifier == 'SVC':
                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                clf = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0).fit(emb_train,
                                                                                                 support_target.detach().numpy())

            test_acc = clf.score(emb_test, query_target.detach().numpy())
            test_acc_all.append(test_acc)

        final_mean = np.mean(test_acc_all)
        final_std = np.std(test_acc_all)
        final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))

        log = {mode+"_acc": final_mean, mode+"_std": final_std, mode+"_interval": final_interval}
        self.log_dict(log,
                      prog_bar=True,
                      batch_size=args.task_num,
                      add_dataloader_idx=False
                      )

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="val")
        elif dataloader_idx == 1:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="t_val")


    def test_step(self, batch, batch_idx):
        self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.max_epochs
        )
        return [optimizer], [lr_scheduler]

@torch.no_grad()
def distributed_sinkhorn(out, epsilon=0.05, num_iterations=3, normalization='col'):
    # https://github.com/facebookresearch/swav/blob/main/main_swav.py

    Q = torch.exp(out / epsilon) # Q is B-by-K (B = batch size, K = queue size)
    B = Q.shape[0] * idist.get_world_size()
    K = Q.shape[1]

    # make the matrix sums to 1
    Q /= idist.all_reduce(torch.sum(Q))

    if normalization == 'col':
        for it in range(num_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

            # normalize each column: total weight per sample must be 1/B
            Q /= idist.all_reduce(torch.sum(Q, dim=0, keepdim=True))
            Q /= K

        Q *= K # the colomns must sum to 1 so that Q is an assignment
    else:
        for it in range(num_iterations):
            # normalize each column: total weight per sample must be 1/B
            Q /= idist.all_reduce(torch.sum(Q, dim=0, keepdim=True))
            Q /= K

            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q


class GFSX(pl.LightningModule):
    def __init__(self,
                 args,
                 data,
                 augmentor,
                 test_idx,
                 encoder_momentum: float = 0.999,
                 encoder_depth=4,
                 head_depth=2,
                 softmax_temperature: float = 0.5,
                 learning_rate: float = 1e-3,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])
        self.args = args
        self.data = data
        self.test_idx = test_idx

        self.aug1, self.aug2, self.aug3 = augmentor

        # create encoders and projection heads
        self.encoder_q, self.encoder_k, self.pretraining_head_q, self.pretraining_head_k = self._init_encoders(args)
        # initialize weights
        self.encoder_q.apply(self._init_weights)
        self.pretraining_head_q.apply(self._init_weights)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            paramh_k.data.copy_(paramh_q.data)  # initialize
            paramh_k.requires_grad = False  # not update by gradient


    def _init_encoders(self, args):
        if args.base_model == 'MLP':
            encoder_q = MLP(args.input_dim, args.out_dim, args.num_layers)
            encoder_k = MLP(args.input_dim, args.out_dim, args.num_layers)
        else:
            encoder_q = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )
            encoder_k = GNNModel(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                layer_name=args.base_model,
                activation_name=args.activation,
                dp_rate=args.dropout
            )

        # Initialize pretraining_head with MLP
        pretraining_head_q = MLP(args.out_dim, args.out_dim)
        pretraining_head_k = MLP(args.out_dim, args.out_dim)

        return encoder_q, encoder_k, pretraining_head_q, pretraining_head_k

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
        for paramh_q, paramh_k in zip(self.pretraining_head_q.parameters(), self.pretraining_head_k.parameters()):
            em = self.hparams.encoder_momentum
            paramh_k.data = paramh_k.data * em + paramh_q.data * (1.0 - em)



    def forward(self,
                data: torch_geometric.data.data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index, edge_weight)
        x3, edge_index3, edge_weight3 = self.aug3(x, edge_index, edge_weight)


        # GNN model
        z1 = self.encoder_q(x1, edge_index1, edge_weight1)
        z1 = self.pretraining_head_q(z1)
        # z1 = self.prediction_head(z1)
        z1 = nn.functional.normalize(z1, dim=1)

        with torch.no_grad():
            z2 = self.encoder_q(x2, edge_index2, edge_weight2)
            #z2 = self.pretraining_head_k(z2)
            z2 = nn.functional.normalize(z2, dim=1)

            z3 = self.encoder_q(x3, edge_index3, edge_weight3)
            # z3 = self.pretraining_head_k(z3)
            z3 = nn.functional.normalize(z3, dim=1)


        return z1, z2, z3

    def _calculate_fs_loss(self, z1, z2, z3, query_idx, queue_mask=None):
        z1_query = z1[query_idx]
        z2_query = z2[query_idx]
        #z3_query = z3[query_idx]

        sim = torch.einsum("nc,bc->nb", [z2_query, z3])
        # if self.args.label_mask or self.args.khop_mask:
        #     sim *= queue_mask

        # randomly select k samples
        # randk_idx = torch.zeros([self.args.n_way, self.args.k_shot], dtype=torch.long)
        # for i in range(self.args.n_way):
        #     true_indices = torch.nonzero(queue_mask[i], as_tuple=True)[0]
        #     shuffled_indices = torch.randperm(true_indices.size(0))
        #     randk_idx[i] = true_indices[shuffled_indices[:self.args.k_shot]]
        # support_embeddings = z3[randk_idx]

        # select according to topk similarity
        topk_idx = torch.topk(sim, k=self.args.k_shot, dim=1, largest=True).indices
        support_embeddings = z3[topk_idx]

        prototypes = support_embeddings.mean(dim=1)

        loss_fs = (torch.mm(z1_query, support_embeddings.reshape(self.args.out_dim, -1)).div(
            self.args.temperature2).logsumexp(dim=1) - z1_query.mul(prototypes).div(self.args.temperature2).sum(
            dim=1)).mean()

        return loss_fs

    def _calculate_label_penalty(self, z1, z2, query_idx):
        z1_query = z1[query_idx]
        sim = torch.einsum("nc,bc->nb", [z1_query, z2])
        sim = normalize_0to1(sim)

        data = self.data
        query_label = data.y[query_idx].view(-1, 1)
        mask = data.y.T == query_label

        loss_penalty = 0
        count = 0
        loss_function = nn.BCELoss()

        for i, idx in enumerate(query_idx):
            if int(idx) in self.test_idx:
                continue
            else:
                #loss_penalty += (sim[i,:] - mask[i, :].to(torch.int32)).norm(dim=0, p=2)
                loss_penalty += loss_function(sim[i, :], mask[i, :].to(torch.float32))
                count += 1

        if count > 0:
            return loss_penalty/count
        else:
            return loss_penalty


    def _calculate_mask(self, query_idx):
        data = self.data
        queue_mask = None
        mask = torch.ones([self.args.n_way, self.args.num_samples])

        # label mask: [n_way, len(queue)]
        if self.args.label_mask:
            query_label = data.y[query_idx].view(-1,1)
            mask = data.y.T == query_label
            neg_mask = data.y.T != query_label
            for i, idx  in enumerate(query_idx):
                if int(idx) in self.test_idx:
                    mask[i, :] = 1

        elif self.args.khop_mask != 0:
            mask = torch.zeros([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                subset, _, _, _ = k_hop_subgraph(int(idx), self.args.khop_mask, data.edge_index)
                mask[row, subset] = 1

        elif self.args.self_mask:
            mask = torch.ones([self.args.n_way, self.args.num_samples])
            for row, idx in enumerate(query_idx):
                mask[row, idx] = 0

        return mask


    def training_step(self, batch, batch_idx):

        self._momentum_update_key_encoder()
        z1, z2, z3 = self(data=self.data.to(batch.device))
        assert z1.requires_grad == True
        assert z2.requires_grad == False
        assert z3.requires_grad == False

        # calculate few-shot loss
        loss_fs1 = 0
        loss_fs2 = 0
        task_num = batch.size()[0] / self.args.n_way
        assert int(task_num) == self.args.train_task_num
        for i in range(int(task_num)):
            query_idx = batch[i*self.args.n_way:i*self.args.n_way+self.args.n_way]
            queue_mask = self._calculate_mask(query_idx).to(batch.device)
            if self.args.compare_mode == 'three':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'z1toz23':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'z23toz1':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z1, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z1, query_idx=query_idx, queue_mask=queue_mask)
            elif self.args.compare_mode == 'm1m2':
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z2, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs1 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z3, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z3, z3=z2, query_idx=query_idx, queue_mask=queue_mask)
                loss_fs2 += self._calculate_fs_loss(z1=z1, z2=z1, z3=z2, query_idx=query_idx, queue_mask=queue_mask)

        loss_fs1 /= task_num
        loss_fs2 /= task_num
        loss_fs = (loss_fs1 + loss_fs2) / 2

        # loss_penalty = self._calculate_label_penalty(z1=z1, z2=z2, query_idx=query_idx)
        # loss_penalty += self._calculate_label_penalty(z1=z2, z2=z3, query_idx=query_idx)
        # loss_fs += 0.5 * loss_penalty

        log = {"train_loss_fs": loss_fs, "train_loss_fs1": loss_fs1, "train_loss_fs2": loss_fs2}
        self.log_dict(log)
        #print(type(loss_fs))

        return loss_fs

    def fs_test(self, batch, data, args, mode="val"):

        task, target = batch

        encoder_model = self.encoder_q
        encoder_model.eval()
        embeddings = encoder_model(data.x,
                                   data.edge_index,
                                   data.edge_attr).detach().cpu().numpy()

        test_acc_all = []
        for i in range(args.task_num):
            task_idx = i * args.n_way
            random_support = torch.randperm(args.n_way * args.k_shot)
            random_query = torch.randperm(args.n_way * args.q_query)

            support_idx = task[task_idx:task_idx + args.n_way, :args.k_shot].reshape(1, -1).squeeze()[random_support].detach().cpu().numpy()
            query_idx = task[task_idx:task_idx + args.n_way, args.k_shot:].reshape(1, -1).squeeze()[random_query].detach().cpu().numpy()

            task_target = target[task_idx:task_idx + args.n_way]
            support_target = map_class(task_target, args.k_shot)[random_support]
            query_target = map_class(task_target, args.q_query)[random_query]

            emb_train = embeddings[support_idx]
            emb_test = embeddings[query_idx]

            if args.classifier == 'LR':
                clf = SKLR(solver='lbfgs', max_iter=1000, multi_class='auto').fit(emb_train,
                                                                              support_target.detach().numpy())
            elif args.classifier == 'SVC':
                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                clf = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0).fit(emb_train,
                                                                                                 support_target.detach().numpy())

            test_acc = clf.score(emb_test, query_target.detach().numpy())
            test_acc_all.append(test_acc)

        final_mean = np.mean(test_acc_all)
        final_std = np.std(test_acc_all)
        final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))

        log = {mode+"_acc": final_mean, mode+"_std": final_std, mode+"_interval": final_interval}
        self.log_dict(log,
                      prog_bar=True,
                      batch_size=args.task_num,
                      add_dataloader_idx=False
                      )

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="val")
        elif dataloader_idx == 1:
            self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="t_val")


    def test_step(self, batch, batch_idx):
        self.fs_test(batch, data=self.data.to(batch[0].device), args=self.args, mode="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.max_epochs
        )
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     self.args.lr,
        #     momentum=self.hparams.momentum,
        #     weight_decay=self.args.weight_decay,
        # )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     self.trainer.max_epochs,
        # )
        return [optimizer], [lr_scheduler]
