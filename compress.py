import io
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset


def preprocess_and_cache(dataset, fmt="JPEG", quality=100):
    """Compress + decode each image ONCE and store as tensors."""
    cached_imgs = []
    cached_labels = []
    to_tensor = T.ToTensor()

    for img, label in dataset:
        # Ensure PIL
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)

        # --- compress ---
        buf = io.BytesIO()
        if fmt in ["JPEG", "WEBP"]:
            img.save(buf, format=fmt, quality=quality)
        else:
            img.save(buf, format=fmt)
        buf.seek(0)

        # --- decode once ---
        comp_img = Image.open(buf).convert("L")
        comp_tensor = to_tensor(comp_img)

        cached_imgs.append(comp_tensor)
        cached_labels.append(label)

    # convert to tensors
    cached_imgs = torch.stack(cached_imgs)
    cached_labels = torch.tensor(cached_labels)

    print(f"Cached {len(dataset)} compressed samples.")

    return torch.utils.data.TensorDataset(cached_imgs, cached_labels)

class CompressedDataset(Dataset):
    def __init__(self, base_dataset, fmt, quality, debug=False):
        self.base = base_dataset
        self.fmt = fmt.upper()
        self.quality = quality
        self.debug = debug
        self.to_tensor = T.ToTensor()

    def compress(self, pil_img):
        buf = io.BytesIO()

        # JPEG/WEBP need quality parameter
        if self.fmt in ["JPEG", "WEBP"]:
            pil_img.save(buf, format=self.fmt, quality=self.quality)
        else:
            pil_img.save(buf, format=self.fmt)

        buf.seek(0)
        comp_img = Image.open(buf).convert("L")  # ensure grayscale

        return comp_img, buf

    def __getitem__(self, idx):
        img, label = self.base[idx]

        # Convert to PIL if needed
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)

        # --- COMPRESS (keep only bytes) ---
        buf = io.BytesIO()
        if self.fmt in ["JPEG", "WEBP"]:
            img.save(buf, format=self.fmt, quality=self.quality)
        else:
            img.save(buf, format=self.fmt)

        # --- NO DECODE ---
        # Just return the original image (or a no-op version)
        return T.ToTensor()(img), label


    def __len__(self):
        return len(self.base)


def apply_compression(dataset, config, debug=False):
    fmt = config["format"]
    quality = config.get("quality", 100)
    return CompressedDataset(dataset, fmt, quality, debug)
