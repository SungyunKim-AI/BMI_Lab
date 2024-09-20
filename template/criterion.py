"""
Reference
ROBUST CURRICULUM LEARNING: FROM CLEAN LABEL DETECTION TO NOISY LABEL SELF-CORRECTION
"""
import numpy as np
import torch
from torch.nn import CrossEntropyLoss


# TODO : 병렬 연산 가능하게 변경 필요
def compute_EMA(losses, decay):
    ema_loss = []
    for i, loss in enumerate(losses):
        if loss is not None:
            ema_loss.append(decay*loss + (1-decay)*ema_loss[i-1])
        else:   # 해당 epoch에서 select 되지 않은 샘플은 이전 epoch의 loss 사용
            ema_loss.append(ema_loss[i-1])
        
    return ema_loss

# EMA loss for each segment
def EMALoss(outputs, labels, decay=0.95, loss_fn=CrossEntropyLoss()):
    instant_loss = loss_fn(outputs, labels)
    return compute_EMA(instant_loss, decay)


# consistency loss for each segment
def EMAConsistencyLoss(outputs, args_outputs, decay=0.95, loss_fn=CrossEntropyLoss()):
    instant_consist_loss = loss_fn(outputs, args_outputs)
    return compute_EMA(instant_consist_loss, decay)
