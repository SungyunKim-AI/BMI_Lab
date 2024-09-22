import os
import torch
import gc
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def get_device(cfg):
    if cfg.gpu_type == 'cpu':
        print("Training on CPU")
        return torch.device('cpu')
    elif cfg.gpu_type == 'single_gpu':
        if torch.cuda.is_available():
            print("Training on single GPU")
            return torch.device('cuda')
        else:
            raise Exception("CUDA is not available.")
    elif cfg.gpu_type == 'multi_gpu':
        if torch.cuda.is_available() and (torch.cuda.device_count() > 1):
            print("Training on multi GPU")
        return 'multi_gpu'
    else:
        raise Exception(f"{cfg.gpu_type} is not in GPU options")

def get_multi_device(cfg):
    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        
        world_size = cfg.n_gpus
        if num_available_gpus < world_size:
            world_size = torch.cuda.device_count()
            print(f'Max world_size is {world_size}')
        
        print(f"Number of GPUs available: {num_available_gpus}")
        for i in range(num_available_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    
        return world_size
    else:
        raise Exception("CUDA is not available.")


# ========== Multi GPU ==========
def setup(rank, world_size, port='12355'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    # 프로세스 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

def cleanup_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    gc.collect()
        
