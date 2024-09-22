import os
import copy
from tqdm import tqdm
from utils.util import *
from utils.gpu import *
from utils.logger import Logger
from dataset import *
from models.nEMGNet import nEMGNet
from models.SimpleCNN import SimpleCNN

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp



class Trainer:
    def __init__(self, dataset, model, optimizer, loss_fn, scheduler, config):
        self.cfg = config
        self.device = get_device(self.cfg)
        
        self.dataset = dataset
        self.model_name = model
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.set_loss_fn(loss_fn)
        
        self.train_losses = AverageMeter()
        self.valid_losses = AverageMeter()
        
        self.epoch = 0
        self.results_path = "./results"
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)
        
        self.logger = Logger(self.cfg)


    def set_model(self, model_name):
        if model_name == 'SimpleCNN':
            self.model = SimpleCNN()
        elif model_name == 'nEMGNet':
            self.model = nEMGNet(n_classes=self.cfg.n_classes)
        else:
            raise Exception(f"{model_name} is not in model dict")
        
    
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
        scaler = GradScaler(device.type)
        with tqdm(dataloader, unit="train_batch", desc=f'Train ({self.epoch}epoch)') as tqdm_loader:
            for step, (X, y) in enumerate(tqdm_loader):
                X = X.to(device)
                y = y.to(device)
                
                self.optimizer.zero_grad()
                with autocast(device.type, enabled=self.cfg.train.AMP):
                    y_preds = self.model(X)
                    loss = self.loss_fn(y_preds, y)
                
                self.train_losses.update(loss.detach().item(), self.cfg.train.batch_size)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                global_step = int(self.epoch * len(dataloader) + step)
                if global_step % self.cfg.log.interval == 0:
                    self.logger.update({"train/step": global_step, "train/loss" : self.train_losses.avg})
        
        print(f'\t\tAvg loss : {self.train_losses.avg}')
                    
        
    
    def training(self, rank, world_size, dataset):
        train_dataloader, valid_dataloader = create_dataloaders(dataset, self.cfg, rank, world_size)
        self.set_model(self.model_name)
        self.set_optimizer(self.optimizer_name)
        self.set_scheduler(self.scheduler_name)
        
        if rank is not None:
            setup(rank, world_size)
            device = rank
            model = DDP(self.model.to(device), device_ids=[device])
        else:
            device = self.device
            model = self.model.to(device)
            rank = 0
        
        best_loss = float('inf')
        for epoch in range(self.cfg.train.epochs):
            self.epoch = epoch
            model.train()
            
            self.train_one_epoch(train_dataloader, device)
            self.scheduler.step()
            
            if (rank == 0) and (epoch % self.cfg.valid.step) == 0:
                valid_loss = self.valid(model, valid_dataloader, device)
                
                if (best_loss > valid_loss) and self.cfg.save_model:
                    torch.save(model.state_dict(), f"{self.results_path}/{self.model_name}_{self.cfg.version}.pth")
                    self.best_model = copy.deepcopy(model)
                    best_loss = valid_loss
                    print(f"Save Best Model : {self.epoch} epoch ({best_loss})")
                    
                self.save_checkpoint(valid_loss)
            
        
        
    def train_KFolds(self):
        self.models = []
        skf = StratifiedKFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=self.cfg.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(self.dataset)), self.dataset.targets)):
            print(f'======== Fold {fold + 1} ========')
            
            train_subset = Subset(self.dataset, train_idx)
            valid_subset = Subset(self.dataset, val_idx)

            if self.cfg.gpu_type == 'multi_gpu':
                world_size = get_multi_device(self.cfg)
                subset = {'train': train_subset, 'valid': valid_subset}
                mp.spawn(self.training, args=(world_size, subset), nprocs=world_size, join=True)
                cleanup()
            else:
                self.training(rank=None, world_size=None, dataset={'train': train_subset, 'valid': valid_subset})
            self.models.append(self.best_model.state_dict())
        
        torch.save(self.models, f"{self.results_path}/ensemble_{self.model_name}_{self.cfg.version}.pth")
        
            
    def train(self):
        results = None
        if self.cfg.n_folds == 1:
            if self.cfg.gpu_type == 'multi_gpu':
                world_size = get_multi_device(self.cfg)
                mp.spawn(self.training, args=(world_size, self.dataset), nprocs=world_size, join=True)
                results = self.best_model
                cleanup()
            else:
                self.training(rank=None, world_size=None, dataset=self.dataset)
                results = self.best_model
        else:
            self.train_KFolds()
            results = self.models
        
        self.logger.finish()
        cleanup_cache()
        return results
    
    def valid(self, model, dataloader, device):
        model.eval()
        preds_dict = {'preds':[], 'labels':[]}
        
        with tqdm(dataloader, unit="valid_batch", desc=f'Valid ({self.epoch}epoch)') as tqdm_loader:
            for step, (X, y) in enumerate(tqdm_loader):
                X, y = X.to(device), y.to(device)
                
                with torch.no_grad():
                    y_preds = model(X)
                    loss = self.loss_fn(y_preds, y)
                
                self.valid_losses.update(loss.detach().item(), self.cfg.valid.batch_size)
                
                preds_dict['preds'].extend(y_preds.detach().tolist())
                preds_dict['labels'].extend(y.detach().tolist())
                
            metrics_macro = compute_metrics(preds_dict, average='macro')
            metrics_weighted = compute_metrics(preds_dict, average='weighted')
            
            log_dict = {
                "vaild/epoch" : self.epoch,
                "valid/loss" : self.valid_losses.avg,
                "valid/acc" : metrics_macro['acc'],
                "valid/confusion_matrix" : metrics_macro['confusion_matrix'],
                "valid/macro/precision" : metrics_macro['precision'],
                "valid/macro/recall" : metrics_macro['recall'],
                "valid/macro/f1" : metrics_macro['f1'],
                "valid/macro/roc_auc" : metrics_macro['roc_auc'],
                "valid/weighted/precision" : metrics_weighted['precision'],
                "valid/weighted/recall" : metrics_weighted['recall'],
                "valid/weighted/f1" : metrics_weighted['f1'],
                "valid/weighted/roc_auc" : metrics_weighted['roc_auc']
            }
            self.logger.update(log_dict)
        
        print(f'\t\tAvg loss : {self.valid_losses.avg}')
        return self.valid_losses.avg

    def save_checkpoint(self, valid_loss):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': valid_loss
        }
        torch.save(checkpoint, f'{self.results_path}/checkpoint.pth')
    
    
    def load_checkpoint(self, path=None):
        if path is None:
            checkpoint = torch.load(f'{self.results_path}/checkpoint.pth')
        else:
            checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Checkpoint loaded from {self.results_path}/checkpoint.pth, resuming from epoch {epoch}")
        return epoch, loss
        