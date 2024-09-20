import os
import argparse
import numpy as np
import pandas as pd
import yaml
from box import Box
from utils import *
from dataset import nEMGDataset, stratified_split
from model import nEMGNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold


def get_config(path):
    with open(path) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
    return config

def train(model):
    model.train()

def valid(model):
    model.eval()

def save_model(model):
    model

def main():
    # ============= Init ============= 
    cfg = get_config('config.yaml')
    set_seed(cfg.seed)
    
    # ============= Data ============= 
    cohort_file = os.path.join(cfg.root_dir, 'cohort_with_anatomy.parquet')
    wav_file = os.path.join(cfg.root_dir, 'wav')

    dataset = nEMGDataset(cohort_file, wav_file)
    train_dataset, test_dataset, loss_weights = stratified_split(dataset, test_size=0.2)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.test.batch_size, shuffle=True)
    
    if cfg.kfolds > 1:
        skf = StratifiedKFold(n_splits=cfg.kfolds, shuffle=True, random_state=cfg.seed)
        
    
    # ============= Hyperparametets ============= 
    model = nEMGNet(n_classes=cfg.n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = StepLR(optimizer, step_size=cfg.train.lr_step, gamma=cfg.train.lr_decay)
    
    # ============= Training ============= 
    results = [0.0]
    for epoch in cfg.train.epochs:
        train(train_dataloader)
        
        if (epoch % cfg.test.step) == 0:
            metric = valid(test_dataloader)
            results.append(metric)
        
            if (results[-2] < results[-1]) and cfg.test.save:
                save_model()
                
        

if __name__ == '__main__':
    main()
    