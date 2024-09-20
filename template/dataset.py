import os
import numpy as np
import pandas as pd
from collections import Counter
from utils import *

import torch
from torch.utils.data import Dataset, Subset


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
    
    # class 별 weight 계산
    train_targets = labels[train_indices]
    class_distribution = Counter(train_targets)
    print('Class Distribution : ', class_distribution)
    total_samples = sum(class_distribution.values())
    class_weights = {cls: total_samples / count for cls, count in class_distribution.items()}
    weights = torch.tensor([class_weights[i] for i in range(len(class_distribution))], dtype=torch.float)
    print('Loss Weights : ', weights.tolist())
    
    
    return Subset(dataset, train_indices), Subset(dataset, test_indices), weights