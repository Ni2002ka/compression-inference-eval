import torch
import torch.nn as nn
import torch.optim as optim
import time


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                
            nn.LazyLinear(256), # Automatically infer input features
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class SmallNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512), # Automatically infer input features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(x)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LazyConv2d(16, 3, padding=1),   # auto infer channels
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(10)    # auto infer input features
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


##### Test functions #####
    # TODO: move this somewhere else


@torch.no_grad()
def test(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0

    start_time = time.time()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    end_time = time.time()
    acc = correct / total
    print(f"Test accuracy: {acc*100:.2f}%")
    print(f"Total testing time: {end_time - start_time:.2f}s")
    return acc, end_time - start_time
