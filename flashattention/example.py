import time
from tqdm import tqdm
import gc
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.cuda import memory_reserved
from Transformer import Transformer


def train(conf, device, dtype):
    # 임의 데이터 생성
    x_train = torch.randn(config['batch_size'], config['seq_len'], config['embed_dim']).to(device=device, dtype=dtype)
    y_train = torch.randint(0, config['num_classes'], (config['batch_size'],), device=device)

    model = Transformer(conf['allow_flash'],
                        embed_dim=conf['embed_dim'],
                        num_heads=conf['num_heads'],
                        ff_hidden_dim=2048,
                        seq_len=conf['seq_len'],
                        num_classes=conf['num_classes'],
                        dropout=0.1).to(device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])

    scaler = GradScaler()
    for epoch in tqdm(range(conf['epochs']), ncols=50):
        model.train()
        optimizer.zero_grad()

        with autocast(device_type=device.type, dtype=dtype):
            outputs = model(x_train) 
            loss = criterion(outputs, y_train) 

        scaler.scale(loss).backward()  
        scaler.step(optimizer)    
        scaler.update()

    print(f"GPU Memory Usage: {memory_reserved(0) / 1024**2:.2f} MB")

def clear_gpu_memory():
    gc.collect() 
    torch.cuda.empty_cache()

if __name__ == "__main__":
    config = {
        'batch_size' : 32,
        'seq_len' : 1024,
        'embed_dim' : 512,
        'num_heads' : 32,
        'num_classes' : 10,
        'lr' : 1e-4,
        'epochs' : 100,
        'allow_flash' : True,
        'dtype': torch.float16, 
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32


    # print("1. Training with FlashAttention...")
    # start_time = time.time()
    # train(config, device, dtype)
    # print(f"Training time : {time.time() - start_time:.2f} seconds", end="\n\n")

    # print("2. Training without FlashAttention...")
    # start_time = time.time()
    # config['allow_flash'] = False
    # train(config, device, dtype)
    # print(f"Training time : {time.time() - start_time:.2f} seconds", end="\n\n")

    print("3. Naive Attention Training...")
    start_time = time.time()
    config['allow_flash'] = False
    train(config, device, dtype)
    print(f"Training time : {time.time() - start_time:.2f} seconds", end="\n\n")

    # clear_gpu_memory()

    

    

    
