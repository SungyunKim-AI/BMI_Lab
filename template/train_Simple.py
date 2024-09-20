import os
import argparse
import numpy as np
import pandas as pd
import yaml
from box import Box
from tqdm import tqdm
from utils import *
from dataset import CustomDataset, stratified_split
from model import nEMGNet

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold


def get_config(path):
    with open(path) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
    return config

def train(cfg, dataloader, model, loss_fn, optimizer, epoch, scheduler, device):
    scaler = GradScaler()
    model.train()
    losses = AverageMeter()
    
    with tqdm(dataloader, unit="train_batch", desc=f'Train ({epoch}epoch)') as tqdm_loader:
        for step, (X, y) in enumerate(tqdm_loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            with autocast(enabled=cfg.AMP):
                y_preds = model(X)
                loss = loss_fn(y_preds, y)
            
            losses.update(loss.detach().item(), cfg.batch_size)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    scheduler.step()
            

def valid(cfg, dataloader, model, loss_fn, epoch, device):
    model.eval()
    losses = AverageMeter()
    preds_dict = {'preds':[], 'labels':[]}
    with tqdm(dataloader, unit="valid_batch", desc=f'Train ({epoch}epoch)') as tqdm_loader:
        for step, (X, y) in enumerate(tqdm_loader):
            X, y = X.to(device), y.to(device)
            
            with torch.no_grad():
                y_preds = model(X)
                loss = loss_fn(y_preds, y)
            
            losses.update(loss.detach().item(), cfg.batch_size)
            preds_dict['preds'].append(y_preds.detach().item())
            preds_dict['labels'].append(y.detach().item())
            

    return losses.avg, preds_dict

def save_model(model):
    model

def main():
    # ============= Init ============= 
    cfg = get_config('config.yaml')
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using', torch.cuda.device_count(), 'GPU(s)')
    
    # ============= Data ============= 
    cohort_file = os.path.join(cfg.root_dir, 'cohort_with_anatomy.parquet')
    wav_file = os.path.join(cfg.root_dir, 'wav')

    # TODO : CustomDataset
    dataset = CustomDataset(cohort_file, wav_file)
    train_dataset, test_dataset, loss_weights = stratified_split(dataset, test_size=0.2)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.test.batch_size, shuffle=True)
    
    # ============= Hyperparametets ============= 
    # TODO : CustomModel
    model = nEMGNet(n_classes=cfg.n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = StepLR(optimizer, step_size=cfg.train.lr_step, gamma=cfg.train.lr_decay)
    
    # ============= Training ============= 
    results = [0.0]
    for epoch in cfg.train.epochs:
        train(cfg.test, train_dataloader, model, loss_fn, optimizer, epoch, scheduler, device)
        
        if (epoch % cfg.test.step) == 0:
            valid_loss, preds_dict = valid(cfg.test, test_dataloader, model, loss_fn, epoch, device)
            results.append(valid_loss)
            
            if (results[-2] < results[-1]) and cfg.test.save:
                save_model()
                
        

if __name__ == '__main__':
    main()
    