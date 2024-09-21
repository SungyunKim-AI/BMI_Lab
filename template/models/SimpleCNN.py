import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3채널 입력, 32채널 출력, 커널 크기 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 32채널 입력, 64채널 출력, 커널 크기 3x3
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling (2x2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 8x8은 MaxPool 적용 후 크기
        self.fc2 = nn.Linear(512, 10)  # CIFAR-10에는 10개의 클래스가 있음

    def forward(self, x):
        # Convolution + ReLU + Max Pooling
        x = self.pool(torch.relu(self.conv1(x)))  # 첫 번째 Conv 레이어
        x = self.pool(torch.relu(self.conv2(x)))  # 두 번째 Conv 레이어
        
        # Flattening (1D로 변환)
        x = x.view(-1, 64 * 8 * 8)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
