import os
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_logger(self.cfg)
    
    def set_logger(self, cfg):
        if cfg.log.platform == 'wandb':
            wandb.init(project=cfg.project, config=cfg)
            wandb.run.name = f'{cfg.project}_{cfg.version}' 
        elif cfg.log.platform == 'tensorboard':
            if not os.path.exists(cfg.log.dir):
                os.mkdir(cfg.log.dir)
            self.writer = SummaryWriter(log_dir=f'{cfg.log.dir}/{cfg.project}_{cfg.version}')
        else:
            raise Exception(f'{cfg.log.platform} is not in Logging Options')
        
    def update(self, log_dict):
        if self.cfg.log.platform == 'wandb':
            wandb.log(log_dict)
        elif self.cfg.log.platform == 'tensorboard':
            
            for key, value in log_dict.items():
                if key in ['train/step', 'vaild/epoch']:
                    global_step = value
                    break
            if 'valid/confusion_matrix' in log_dict:
                log_dict.pop('valid/confusion_matrix')
                    
            for key, value in log_dict.items():
                self.writer.add_scalar(key, value, global_step)
            
    
    def finish(self):
        if self.cfg.log.platform == 'wandb':
            wandb.finish()
        elif self.cfg.log.platform == 'tensorboard':
            self.writer.close()