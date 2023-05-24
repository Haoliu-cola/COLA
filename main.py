# Standard libraries
import os

# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError

# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# PyTorch geometric
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from torch.utils.data import DataLoader, TensorDataset
import copy

# PL callbacks
import wandb
import lightning.pytorch as pl

from torch import Tensor
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities import model_summary
import GCL.augmentors as A
from GCL.eval import LREvaluator

from model_interface import GFS, GFS2,GFS3, GFS4, GFS5


from data import load_dataset
from utils import get_args

from data import FewShotDataManager, IndexDataset, IndexDataset3

def train_gfs(args,
              max_epochs,
              dataset):
    project_name = '-'.join([args.exp_name, args.dataset, str(args.n_way), str(args.k_shot)])
    logger = WandbLogger(name=args.name, project=project_name, save_dir=args.save_dir)
    logger.log_hyperparams(args)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger,
        min_epochs=100,
        log_every_n_steps=1,
        val_check_interval=0.2,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            ModelCheckpoint(save_weights_only=True, dirpath=args.save_dir + '/' + args.dataset, monitor="val_acc",
                            mode="max", save_top_k=3),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_acc", min_delta=0.00, patience=200, verbose=False, mode="max")
        ]
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    pretrained_filename = args.best_pretrain
    print(pretrained_filename)
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading... ")
        model = GFS.load_from_checkpoint(pretrained_filename)

    else:
        pl.seed_everything(args.random_seed)
        train_loader = DataLoader(IndexDataset(dataset[0]),
                                  batch_size=args.n_way * args.train_task_num,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_workers)


        datamanager = FewShotDataManager(dataset[0], args)
        t_val_loader = datamanager.get_data_loader(0)
        val_loader = datamanager.get_data_loader(1)
        test_loader = datamanager.get_data_loader(2)
        test_idx = datamanager.split['test']

        aug1 = A.Compose([A.EdgeRemoving(pe=args.e1), A.FeatureMasking(pf=args.f1)])
        aug2 = A.Compose([A.EdgeRemoving(pe=args.e2), A.FeatureMasking(pf=args.f2)])

        model = GFS(augmentor=(aug1, aug2),
                    args=args,
                    data = dataset[0],
                    test_idx=test_idx
        )


        #trainer.fit(model, train_dataloaders=train_loader)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_loader,test_loader])
        # Load best checkpoint after training
        model = GFS.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                         data=dataset[0])

        trainer.test(model, dataloaders=test_loader)
        trainer.test(model, dataloaders=val_loader)


    wandb.finish()
    return model

def train_gfs2(args,
               max_epochs,
               dataset):
    project_name = '-'.join([args.exp_name, args.dataset, str(args.n_way), str(args.k_shot)])
    logger = WandbLogger(name=args.name, project=project_name, save_dir=args.save_dir)
    logger.log_hyperparams(args)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger,
        min_epochs=50,
        log_every_n_steps=1,
        val_check_interval=0.2,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            ModelCheckpoint(save_weights_only=True, dirpath=args.save_dir + '/' + args.dataset, monitor="val_acc",
                            mode="max", save_top_k=3),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_acc", min_delta=0.00, patience=200, verbose=False, mode="max")
        ]
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    pretrained_filename = args.best_pretrain+'aaaaa'
    print(pretrained_filename)
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading... ")
        model = GFS.load_from_checkpoint(pretrained_filename)

    else:
        pl.seed_everything(args.random_seed)
        train_loader = DataLoader(IndexDataset(dataset[0]),
                                  batch_size=args.n_way * args.train_task_num,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_workers)


        datamanager = FewShotDataManager(dataset[0], args)
        t_val_loader = datamanager.get_data_loader(0)
        val_loader = datamanager.get_data_loader(1)
        test_loader = datamanager.get_data_loader(2)

        aug1 = A.Compose([A.EdgeRemoving(pe=args.e1), A.FeatureMasking(pf=args.f1)])
        aug2 = A.Compose([A.EdgeRemoving(pe=args.e2), A.FeatureMasking(pf=args.f2)])
        aug3 = A.Compose([A.EdgeRemoving(pe=args.e3), A.FeatureMasking(pf=args.f3)])

        model = GFS5(augmentor=(aug1, aug2, aug3),
                     args=args,
                     data = dataset[0]
        )


        #trainer.fit(model, train_dataloaders=train_loader)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_loader,test_loader])
        # Load best checkpoint after training
        model = GFS5.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                         data=dataset[0])

        trainer.test(model, dataloaders=test_loader)
        trainer.test(model, dataloaders=val_loader)


    wandb.finish()
    return model

def train_gfs3(args,
               max_epochs,
               dataset):
    project_name = '-'.join([args.exp_name, args.dataset, str(args.n_way), str(args.k_shot)])
    logger = WandbLogger(name=args.name, project=project_name, save_dir=args.save_dir)
    logger.log_hyperparams(args)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger,
        min_epochs=50,
        log_every_n_steps=1,
        val_check_interval=0.2,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            ModelCheckpoint(save_weights_only=True, dirpath=args.save_dir + '/' + args.dataset, monitor="val_acc",
                            mode="max", save_top_k=3),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_acc", min_delta=0.00, patience=200, verbose=False, mode="max")
        ]
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    pretrained_filename = args.best_pretrain+'aaaaa'
    print(pretrained_filename)
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading... ")
        model = GFS.load_from_checkpoint(pretrained_filename)

    else:
        pl.seed_everything(args.random_seed)

        datamanager = FewShotDataManager(dataset[0], args)
        t_val_loader = datamanager.get_data_loader(0)
        val_loader = datamanager.get_data_loader(1)
        test_loader = datamanager.get_data_loader(2)
        test_idx = datamanager.split['test']

        val_idx = datamanager.split['valid']
        # test_val_idx = test_idx.extend(val_idx)

        if args.label_mask == 3:
            train_loader = DataLoader(IndexDataset3(dataset[0], test_idx),
                                      batch_size=args.n_way * args.train_task_num,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=args.num_workers)
        else:
            train_loader = DataLoader(IndexDataset(dataset[0]),
                                      batch_size=args.n_way * args.train_task_num,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=args.num_workers)
            # train_loader = DataLoader(IndexDataset3(dataset[0], test_idx),
            #                           batch_size=args.n_way * args.train_task_num,
            #                           shuffle=True,
            #                           drop_last=True,
            #                           num_workers=args.num_workers)

        aug1 = A.Compose([A.EdgeRemoving(pe=args.e1), A.FeatureMasking(pf=args.f1)])
        aug2 = A.Compose([A.EdgeRemoving(pe=args.e2), A.FeatureMasking(pf=args.f2)])
        aug3 = A.Compose([A.EdgeRemoving(pe=args.e3), A.FeatureMasking(pf=args.f3)])

        model = GFS3(augmentor=(aug1, aug2, aug3),
                     args=args,
                     data = dataset[0],
                     test_idx = test_idx,
                     encoder_momentum=args.mmt
        )


        #trainer.fit(model, train_dataloaders=train_loader)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_loader,test_loader])
        # Load best checkpoint after training
        model = GFS3.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                         data=dataset[0])

        trainer.test(model, dataloaders=test_loader)
        trainer.test(model, dataloaders=val_loader)


    wandb.finish()
    return model

def main():
    args = get_args()
    dataset = load_dataset(args)

    #
    if args.model_mode == 'fs1':
        GFS_model = train_gfs(args=args, max_epochs=args.max_epochs, dataset=dataset)
    if args.model_mode == 'fs2':
        GFS_model = train_gfs2(args=args, max_epochs=args.max_epochs, dataset=dataset)
    if args.model_mode == 'fs3':
        GFS_model = train_gfs3(args=args, max_epochs=args.max_epochs, dataset=dataset)


if __name__ == "__main__":
    main()