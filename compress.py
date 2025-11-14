import io
import time
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

#############################
#   COMPRESSION UTILITIES   #
#############################

def compress_image(pil_img, format="JPEG", quality=50):
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def apply_compression(dataset, compression_config):
    """Return compressed version of dataset (lazy eval via transform)."""
    format = compression_config["format"]
    quality = compression_config.get("quality", None)

    def compress_fn(img):
        return compress_image(img, format=format, quality=quality)

    compressed_dataset = datasets.ImageFolder(
        dataset.root,
        transform=T.Compose([compress_fn, T.ToTensor()])
    )
    return compressed_dataset

