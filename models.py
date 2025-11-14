import torch
import torch.nn as nn
import torch.optim as optim
import time


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


##### Train/Test functions #####

def train(model, train_loader, device="cpu", epochs=5):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s\n")


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
    return acc
