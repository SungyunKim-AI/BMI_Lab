import os
import numpy as np
import pandas as pd
from collections import Counter
from utils.util import *

import torch
from torch.utils.data import Dataset, DataLoader, Subset, distributed
from sklearn.model_selection import StratifiedKFold


class CustomDataset(Dataset):
    def __init__(self, cohort_file, root_dir, transform=None):
        """
        Arguments:
            cohort_file (string): cohort file name to the parquet file with annotations.
            root_dir (string): Directory with all the signals.
            split (string) : train / valid / test
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {'nl':0, 'rd':1, 'fn':2, 'mn':3, 'pn':4, 'bp':5, 'sm':6}
        
        self.cohort = pd.read_parquet(cohort_file, engine='pyarrow')

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        signal_path = os.path.join(self.root_dir, self.cohort.loc[idx, 'file_name'])
        signal = load_pickle(signal_path)
        label = self.cohort.loc[idx, 'label'].apply(lambda x : self.label_map[x])
        
        sample = {'signal': signal, 'label':label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

# 데이터 분포에 맞게, train / test split
def stratified_split(dataset, test_size=0.2):
    labels = np.array(dataset.label)
    num_classes = len(np.unique(labels))
    
    train_indices, test_indices = [], []
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)
        
        split_idx = int(len(class_indices) * (1 - test_size))
        train_indices.extend(class_indices[:split_idx])
        test_indices.extend(class_indices[split_idx:])

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def get_weights(dataset):
    train_targets = np.array(dataset.label)
    class_distribution = Counter(train_targets)
    print('Class Distribution : ', class_distribution)
    
    total_samples = sum(class_distribution.values())
    class_weights = {cls: total_samples / count for cls, count in class_distribution.items()}
    weights = torch.tensor([class_weights[i] for i in range(len(class_distribution))], dtype=torch.float)
    print('Loss Weights : ', weights.tolist())


def create_dataloaders(dataset, cfg, rank=None, world_size=None):
    if cfg.n_folds > 1:
        train_dataset = dataset['train']
        valid_dataset = dataset['valid']
    else:
        train_dataset, valid_dataset = stratified_split(dataset, test_size=cfg.data_split)
    
    train_loader_params = {
        'batch_size': cfg.train.batch_size,
        'shuffle': True
    }
    
    valid_loader_params = {
        'batch_size': cfg.test.batch_size,
        'shuffle': False,
    }
    
    if cfg.gpu_type in ['single_gpu', 'multi_gpu']:
        params = {'num_workers': cfg.num_workers, 'pin_memory': True, 'non_blocking': True}
        train_loader_params.update(params)
        valid_loader_params.update(params)
        
    if cfg.gpu_type == 'multi_gpu':
        sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        train_loader_params['sampler'] = sampler
        valid_loader_params['sampler'] = sampler
        
    
    train_dataloader = DataLoader(train_dataset, **train_loader_params)
    valid_dataloader = DataLoader(valid_dataset, **valid_loader_params)

    return train_dataloader, valid_dataloader
