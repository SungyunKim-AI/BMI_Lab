import os
import random
import pickle
import numpy as np
import torch

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


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
