import os
import random
import pickle
import yaml
import torch
import numpy as np
from box import Box

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_config(self, path):
    with open(path) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
    return config

def set_seed(seed: int):
    random.seed(seed)           # Python의 기본 랜덤 시드 설정
    np.random.seed(seed)        # NumPy의 랜덤 시드 설정
    torch.manual_seed(seed)     # PyTorch의 랜덤 시드 설정 (CPU)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    
    # PyTorch의 랜덤 시드 설정 (GPU, CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 여러 GPU를 사용하는 경우
    
    # PyTorch에서 연산의 결정성을 보장하려면 다음 옵션을 추가할 수 있습니다.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_metrics(preds_dict, average='macro'):
    y_pred_proba = F.softmax(preds_dict['preds'], dim=1).numpy()
    y_preds = np.array(preds_dict['preds'])
    y = np.array(preds_dict['labels'])
    
    precision = precision_score(y, y_preds, average=average)
    recall = recall_score(y, y_preds, average=average)
    f1 = f1_score(y, y_preds, average=average)
    acc = accuracy_score(y, y_preds)
    
    y_true_onehot = np.eye(y_pred_proba.shape[1])[y]
    roc_auc = roc_auc_score(y_true_onehot, y_pred_proba, average=average, multi_class='ovr')
    
    conf_matrix = confusion_matrix(y, y_preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'acc': acc, 
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix
    }
    