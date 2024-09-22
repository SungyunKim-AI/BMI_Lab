from utils.util import *
from dataset import CustomDataset
from trainer import Trainer

def main():
    cfg = get_config('config.yaml')
    set_seed(cfg.seed)
    
    cohort_file = 'FILE PATH'
    wav_file = 'FOLDER PATH'

    dataset = CustomDataset(cohort_file, wav_file)
    
    trainer = Trainer(dataset, 
                      model = 'nEMGNet', 
                      optimizer = 'Adam', 
                      loss_fn = 'CrossEntropyLoss', 
                      scheduler = 'StepLR', 
                      config=cfg)
    
    best_model = trainer.train()
    
    
if __name__ == '__main__':
    main()
    