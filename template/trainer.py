import os
import copy
from tqdm import tqdm
import wandb
from utils.util import *
from utils.gpu import *
from dataset import *
from models.nEMGNet import nEMGNet
from models.SimpleCNN import SimpleCNN

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, distributed
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

class Trainer:
    def __init__(self, dataset, model, optimizer, loss_fn, scheduler, config):
        self.cfg = config
        self.device = get_device(self.cfg)
        
        self.dataset = dataset
        self.set_model(model)
        self.set_optimizer(optimizer)
        self.set_loss_fn(loss_fn)
        self.set_scheduler(scheduler)
        
        self.train_losses = AverageMeter()
        self.valid_losses = AverageMeter()
        self.preds = {'preds':[], 'labels':[]}
        
        self.epoch = 0
        self.checkpoint_path = "results/checkpoint.pth"
        
        wandb.init(project=self.cfg.wandb.project, config=self.cfg)


    def set_model(self, model):
        self.model_name = model
        if model == 'SimpleCNN':
            self.model = SimpleCNN()
        elif model == 'nEMGNet':
            self.model = nEMGNet(n_classes=self.cfg.n_classes)
        else:
            raise Exception(f"{model} is not in model dict")
    
    def set_optimizer(self, optimizer):
        if optimizer == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.cfg.train.lr)
        else:
            raise Exception(f"{optimizer} is not in optimizer dict")
    
    def set_loss_fn(self, loss_fn):
        if loss_fn == 'CrossEntropyLoss':
            if self.cfg.train.weighted_loss == True:
                loss_weights = get_weights(self.dataset)
                self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise Exception(f"{loss_fn} is not in criterion dict")
    
    def set_scheduler(self, schdeduler):
        if schdeduler == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=self.cfg.train.lr_step, gamma=self.cfg.train.lr_decay)
        else:
            raise Exception(f"{schdeduler} is not in schdeduler dict")

    
    def train_one_epoch(self, dataloader, device):
        scaler = GradScaler()
        with tqdm(dataloader, unit="train_batch", desc=f'Train ({self.epoch}epoch)') as tqdm_loader:
            for step, (X, y) in enumerate(tqdm_loader):
                X = X.to(device)
                y = y.to(device)
                
                self.optimizer.zero_grad()
                with autocast(enabled=self.cfg.train.AMP):
                    y_preds = self.model(X)
                    loss = self.loss_fn(y_preds, y)
                
                logging_loss = loss.detach().item()
                self.train_losses.update(logging_loss, self.cfg.batch_size)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
        
                wandb.log({"train/loss" : logging_loss})
        
        
    
    def training(self, rank, world_size, dataset):
        train_dataloader, valid_dataloader = create_dataloaders(dataset, self.cfg, rank, world_size)
        
        if rank is not None:
            setup(rank, world_size)
            device = rank
            model = DDP(self.model.to(device), device_ids=[device])
        else:
            device = self.device
            model = self.model.to(device)
            rank = 0
        
        best_loss = 1.0
        for epoch in range(self.cfg.train.epochs):
            self.epoch = epoch
            model.train()
            
            self.train_one_epoch(train_dataloader, device)
            self.scheduler.step()
            
            
            if (rank == 0) and (epoch % self.cfg.valid.step) == 0:
                valid_loss = self.valid(model, valid_dataloader, device)
                
                if (best_loss > valid_loss) and self.cfg.save_model:
                    torch.save(model.state_dict(), f"results/{self.model_name}_{self.cfg.version}.pth")
                    self.best_model = copy.deepcopy(model)
                    best_loss = valid_loss
                    
                self.save_checkpoint(valid_loss)
            
        
        
    def train_KFolds(self):
        self.models = []
        skf = StratifiedKFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=self.cfg.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(self.dataset)), self.dataset.label)):
            print(f'======== Fold {fold + 1} ========')
            
            train_subset = Subset(self.dataset, train_idx)
            valid_subset = Subset(self.dataset, val_idx)

            if self.cfg.gpu_type == 'multi_gpu':
                world_size = get_multi_device(self.cfg)
                subset = {'train': train_subset, 'valid': valid_subset}
                mp.spawn(self.training, args=(world_size, subset), nprocs=world_size, join=True)
                wandb.finish()
                cleanup()
            else:
                self.training(rank=None, world_size=None, dataset={'train': train_subset, 'valid': valid_subset})
            self.models.append(self.best_model.state_dict())
        
        torch.save(self.models, f"results/ensemble_{self.model_name}_{self.cfg.version}.pth")
        
            
    def train(self):
        if self.cfg.n_folds == 1:
            if self.cfg.gpu_type == 'multi_gpu':
                world_size = get_multi_device(self.cfg)
                mp.spawn(self.training, args=(world_size, self.dataset), nprocs=world_size, join=True)
                wandb.finish()
                cleanup()
                return self.best_model
            else:
                self.training(rank=None, world_size=None, dataset=self.dataset)
                return self.best_model
        else:
            self.train_KFolds()
            return self.models
    
    def valid(self, model, dataloader, device):
        model.eval()
        
        with tqdm(dataloader, unit="valid_batch", desc=f'Valid ({self.epoch}epoch)') as tqdm_loader:
            for step, (X, y) in enumerate(tqdm_loader):
                X, y = X.to(device), y.to(device)
                
                with torch.no_grad():
                    y_preds = model(X)
                    loss = self.loss_fn(y_preds, y)
                
                logging_loss = loss.detach().item()
                self.valid_losses.update(logging_loss, self.cfg.valid.batch_size)
                self.preds_dict['preds'].extend(y_preds.detach().item())
                self.preds_dict['labels'].extend(y.detach().item())
                
            metrics_macro = compute_metrics(self.preds_dict, average='macro')
            metrics_weighted = compute_metrics(self.preds_dict, average='weighted')
            
            wandb.log({
                "valid/loss" : logging_loss,
                "valid/acc" : metrics_macro['acc'],
                "valid/confusion_matrix" : metrics_macro['confusion_matrix'],
                "valid/macro/precision" : metrics_macro['precision'],
                "valid/macro/recall" : metrics_macro['recall'],
                "valid/macro/f1" : metrics_macro['f1'],
                "valid/macro/roc_auc" : metrics_macro['roc_auc'],
                "valid/macro/precision" : metrics_weighted['precision'],
                "valid/macro/recall" : metrics_weighted['recall'],
                "valid/macro/f1" : metrics_weighted['f1'],
                "valid/macro/roc_auc" : metrics_weighted['roc_auc']
            })
        
        return self.valid_losses.avg

    def save_checkpoint(self, valid_loss):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': valid_loss
        }
        torch.save(checkpoint, self.checkpoint_path)
    
    
    def load_checkpoint(self, path=None):
        if path is None:
            checkpoint = torch.load(self.checkpoint_path)
        else:
            checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Checkpoint loaded from {self.checkpoint_path}, resuming from epoch {epoch}")
        return epoch, loss
        