import io
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset


class CompressedDataset(Dataset):
    def __init__(self, base_dataset, fmt, quality):
        self.base = base_dataset
        self.fmt = fmt
        self.quality = quality
        self.to_tensor = T.ToTensor()

    def compress(self, pil_img):
        buf = io.BytesIO()
        if self.fmt in ["JPEG", "WEBP"]:
            pil_img.save(buf, format=self.fmt, quality=self.quality)
        else:
            pil_img.save(buf, format=self.fmt)
        buf.seek(0)
        comp_img = Image.open(buf).convert("L")
        return comp_img, buf

    def __getitem__(self, idx):        
        img, label = self.base[idx]

        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)

        comp_img, buf = self.compress(img)

        return self.to_tensor(comp_img), label

    def __len__(self):
        return len(self.base)


def apply_compression(dataset, config):
    fmt = config["format"]
    quality = config.get("quality", 100)
    return CompressedDataset(dataset, fmt, quality)
