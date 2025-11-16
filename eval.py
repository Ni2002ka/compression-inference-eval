import csv
import os
import time
import torch


from get_data import get_train_test_data
from models import SmallCNN, SmallMLP, SmallNN
from config import compression_settings, model_list, datasets_list


def load_model_and_hist(checkpoint_dir, model_str, dataset, compressor_name, device="cpu"):
    checkpoint_path = f"{checkpoint_dir}/{model_str}_{dataset}_{compressor_name}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_str == "small-MLP":
        model = SmallMLP()
    elif model_str == "small-NN":
        model = SmallNN()
    elif model_str == "small-CNN":
        model = SmallCNN()
    else:
        raise ValueError(f"Unknown model type: {model_str}")

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Load training history
    history = checkpoint.get("history", {})
    return model, history


@torch.no_grad()
def test(model, test_loader, device="cpu", verbose=False):
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
    if verbose:
        print(f"Test accuracy: {acc*100:.2f}%")
        print(f"Total testing time: {end_time - start_time:.2f}s")
    return acc, end_time - start_time


def load_and_eval_model(model_str="small-MLP", dataset="fashion", download=False, compressor=None):

    model, train_hist = load_model_and_hist("models", model_str=model_str, dataset=dataset, compressor_name="none", device="cpu")
    _, test_loader = get_train_test_data(dataset=dataset, root="data/", train=False, test=True, batch_size=64, download=download, compressor=compressor)
    accuracy, test_time = test(model, test_loader, verbose=True)

    return train_hist, accuracy, test_time



def create_eval_csv():
    csv_filename = "results.csv"

    # Write header row
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "dataset", "compression",
            "avg_epoch_time", "train_loss",
            "test_accuracy", "test_time"
        ])

    for model_name in model_list:
        for dataset_name in datasets_list:
            for compression_name, compression_params in compression_settings.items():

                print(f"\n=== Model: {model_name} | Dataset: {dataset_name} | Compression: {compression_name} ===")

                train_hist, accuracy, test_time = load_and_eval_model(
                    model_str=model_name,
                    dataset=dataset_name,
                    download=True,
                    compressor=compression_params,
                )
                avg_time_per_epoch = train_hist["avg_time_per_epoch"]
                train_loss = train_hist["epoch_losses"][-1]

                # Write results row
                with open(csv_filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        model_name,
                        dataset_name,
                        compression_name,
                        f"{avg_time_per_epoch:.4f}",
                        f"{train_loss:.4f}",
                        f"{accuracy:.4f}",
                        f"{test_time:.4f}",
                    ])

    print("\n*** DONE! Results saved to results.csv ***")

create_eval_csv()