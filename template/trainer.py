import os
import copy
import yaml
from box import Box
from tqdm import tqdm
from utils import *
from dataset import stratified_split
from model import nEMGNet

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold

class Trainer:
    def __init__(self, dataset, model, optimizer, loss_fn, scheduler, config):
        self.cfg = self.get_config(config)
        self.set_device()
        
        self.set_dataloader(dataset)
        self.set_model(model)
        self.set_optimizer(optimizer)
        self.set_loss_fn(loss_fn)
        self.set_scheduler(scheduler)
        
        self.train_losses = AverageMeter()
        self.valid_losses = AverageMeter()
        self.preds = {'preds':[], 'labels':[]}
        
        self.epoch = 0
        self.checkpoint_path = "results/checkpoint.pth"
        self.model_save_path = f"results/{model}_{self.cfg.version}.pth"

    def get_config(self, path):
        with open(path) as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
            config = Box(config_yaml)
        return config
    
    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using', torch.cuda.device_count(), 'GPU(s)')

    def set_dataloader(self, dataset):
        self.train_dataset, self.test_dataset, self.loss_weights = stratified_split(dataset, test_size=self.cfg.data_split)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.train.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.cfg.test.batch_size, shuffle=True)

    def set_model(self, model):
        if model == 'nEMGNet':
            self.optimizer = nEMGNet(n_classes=self.cfg.n_classes).to(self.device)
        else:
            raise Exception(f"{model} is not in model dict")
    
    def set_optimizer(self, optimizer):
        if optimizer == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.cfg.train.lr)
        else:
            raise Exception(f"{optimizer} is not in optimizer dict")
    
    def set_loss_fn(self, loss_fn):
        if loss_fn == 'CrossEntropyLoss':
            self.loss_fn = nn.CrossEntropyLoss(weight=self.loss_weights)
        else:
            raise Exception(f"{loss_fn} is not in criterion dict")
    
    def set_scheduler(self, schdeduler):
        if schdeduler == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=self.cfg.train.lr_step, gamma=self.cfg.train.lr_decay)
        else:
            raise Exception(f"{schdeduler} is not in schdeduler dict")

    
    def train_one_epoch(self, X, y, scaler):
        X = X.to(self.device)
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        with autocast(enabled=self.cfg.train.AMP):
            y_preds = self.model(X)
            loss = self.loss_fn(y_preds, y)
        
        self.train_losses.update(loss.detach().item(), self.cfg.batch_size)
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
    
    def train(self):
        results = [0.0]
        for epoch in self.cfg.train.epochs:
            scaler = GradScaler()
            self.model.train()
            
            with tqdm(self.train_dataloader, unit="train_batch", desc=f'Train ({epoch}epoch)') as tqdm_loader:
                for step, (X, y) in enumerate(tqdm_loader):
                    self.train_one_epoch(X,y, scaler)
                self.scheduler.step()
            
            if (epoch % self.cfg.test.step) == 0:
                valid_loss = self.valid(epoch)
                results.append(valid_loss)
                
                if (results[-2] < results[-1]) and self.cfg.test.save:
                    torch.save(self.model.state_dict(), self.model_save_path)
                    self.best_model = copy.deepcopy(self.model)
                    
            self.save_checkpoint(epoch, valid_loss)
        
        return self.best_model
    
    
    
    def valid(self, epoch):
        self.model.eval()
        with tqdm(self.test_dataloader, unit="valid_batch", desc=f'Valid ({epoch}epoch)') as tqdm_loader:
            for step, (X, y) in enumerate(tqdm_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                
                with torch.no_grad():
                    y_preds = self.model(X)
                    loss = self.loss_fn(y_preds, y)
                
                self.valid_losses.update(loss.detach().item(), self.cfg.test.batch_size)
                self.preds_dict['preds'].append(y_preds.detach().item())
                self.preds_dict['labels'].append(y.detach().item())
                

    def save_checkpoint(self, epoch, valid_loss):
        checkpoint = {
            'epoch': epoch,
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
        