# NOTE: THIS FILE IS BROKEN RN!

import csv
import torchvision.transforms as T
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from get_data import get_train_test_data
from models import SmallCNN, SmallMLP, SmallNN, train, test

#############################
#        MAIN PIPELINE      #
#############################


def run_train_test_pipeline(model_str="small-CNN", dataset="fashion", download=False, compressor=None):

    if model_str == "small-MLP":
        model = SmallMLP()
    elif model_str == "small-NN":
        model = SmallNN()
    elif model_str == "small-CNN":
        model = SmallCNN()
    else:
        raise ValueError(f"Unknown model '{model_str}'. Choose 'small-MLP', 'small-NN', or 'small-CNN'.")


    train_loader, test_loader = get_train_test_data(dataset=dataset, root="data/", batch_size=64, download=download, compressor=compressor)
    avg_time_per_epoch, train_loss = train(model, train_loader, device="cpu", epochs=50)

    accuracy, test_time = test(model, test_loader)

    return avg_time_per_epoch, train_loss, accuracy, test_time



def run_all():
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

                avg_time_per_epoch, train_loss, accuracy, test_time = run_train_test_pipeline(
                    model_str=model_name,
                    dataset=dataset_name,
                    download=True,
                    compressor=compression_params,
                )

                print(f"Avg Time/Epoch: {avg_time_per_epoch:.2f}s | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Test Accuracy: {accuracy*100:.2f}% | "
                    f"Test Time: {test_time:.2f}s")

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


def single_run():
    model_name = "small-CNN"
    dataset_name = "cifar10"
    compression_name = "none"
    compression_params = compression_settings[compression_name]

    print(f"\n=== Model: {model_name} | Dataset: {dataset_name} | Compression: {compression_name} ===")

    avg_time_per_epoch, train_loss, accuracy, test_time = run_train_test_pipeline(
        model_str=model_name,
        dataset=dataset_name,
        download=False,
        compressor=compression_params,
    )

    print(f"Avg Time/Epoch: {avg_time_per_epoch:.2f}s | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test Accuracy: {accuracy*100:.2f}% | "
        f"Test Time: {test_time:.2f}s")
    

single_run()