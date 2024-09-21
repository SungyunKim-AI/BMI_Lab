import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 프로세스 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(rank, world_size):
    setup(rank, world_size)
    
    # 데이터셋 및 데이터 로더 생성
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    # DistributedSampler를 사용하여 데이터셋을 각 GPU에 분배
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=16)
    
    # 모델, 손실 함수, 옵티마이저 정의
    model = SimpleModel().to(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 학습 루프
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # 10 mini-batch마다 출력
                print(f"[{epoch + 1}, {i + 1}] rank {rank} loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    cleanup()

def main():
    world_size = 4
    # 각 GPU에서 프로세스 시작
    start = time.time()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    end = time.time()
    print(f"{end - start:.5f} sec")

if __name__ == "__main__":
    main()
