import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

from config import compression_settings, model_list, datasets_list
from get_data import get_train_test_data
from models import SmallCNN, SmallMLP, SmallNN

def train(model, train_loader, device="cpu", epochs=5, save_path=None, run_name=None, verbose=False):

    if save_path:
        if run_name is None:
            raise ValueError("run_name must be provided when using save_path as a directory.")
        
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {
        "epoch_losses": [],
        "epoch_times": [],
        "total_time": None,
        "avg_time_per_epoch": None,
    }

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

        history["epoch_losses"].append(avg_loss)
        history["epoch_times"].append(epoch_time)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time
    history["total_time"] = total_time
    history["avg_time_per_epoch"] = total_time / epochs

    if verbose:
        print(f"\nTotal training time: {total_time:.2f}s\n")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        checkpoint_path = os.path.join(save_path, f"{run_name}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "hyperparams": {
                "lr": 1e-3,
                "optimizer": "Adam",
                "criterion": "CrossEntropyLoss",
                "batch_size": train_loader.batch_size,
            }
        }, checkpoint_path)

        if verbose:
            print(f"Model checkpoint saved to {checkpoint_path}")

    return history


def train_and_save(model_str="small-CNN", dataset="fashion", download=False, compressor=None, compressor_name=None, save_path="checkpoints/"):

    if model_str == "small-MLP":
        model = SmallMLP()
    elif model_str == "small-NN":
        model = SmallNN()
    elif model_str == "small-CNN":
        model = SmallCNN()
    else:
        raise ValueError(f"Unknown model '{model_str}'. Choose 'small-MLP', 'small-NN', or 'small-CNN'.")

    train_loader, _ = get_train_test_data(dataset=dataset, root="data/", batch_size=64, train=True, test=False, download=download, compressor=compressor)
    run_name = f"{model_str}_{dataset}_{compressor_name}"
    history = train(model, train_loader, device="cpu", epochs=20, save_path=save_path, run_name=run_name)

    return history


for model_name in model_list:
    for dataset_name in datasets_list:
        for compression_name, compression_params in compression_settings.items():

            print(f"\n=== Model: {model_name} | Dataset: {dataset_name} | Compression: {compression_name} ===")

            train_and_save(
                model_str=model_name,
                dataset=dataset_name,
                download=False,
                compressor=compression_params,
                compressor_name=compression_name,
                save_path="models/",
            )
