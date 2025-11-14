
import torchvision.transforms as T
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from get_data import get_train_test_data
from models import SmallCNN, train, test

#############################
#        MAIN PIPELINE      #
#############################

compression_settings = {
    "jpeg_1":   {"format": "JPEG", "quality": 1},
    "jpeg_2":   {"format": "JPEG", "quality": 2},
    "jpeg_5":   {"format": "JPEG", "quality": 5},
    "jpeg_10":  {"format": "JPEG", "quality": 10},
    "jpeg_50": {"format": "JPEG", "quality": 50},
    "webp_20": {"format": "WEBP", "quality": 20},
    "png": {"format": "PNG"},
    "webp_1": {"format": "WEBP", "quality": 1},
    "webp_5": {"format": "WEBP", "quality": 5},
}

def run_train_test_pipeline(model_str="small-CNN", download=False, compressor=None):
    model = SmallCNN()

    train_loader, test_loader = get_train_test_data(dataset="mnist", root="data/", batch_size=64, download=download, compressor=compressor)
    train(model, train_loader, device="cpu", epochs=5)

    accuracy = test(model, test_loader)


run_train_test_pipeline(compressor=compression_settings["jpeg_1"])