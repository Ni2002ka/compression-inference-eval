import io
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset



def preprocess_and_cache(dataset, fmt="JPEG", quality=100):
    cached_imgs = []
    cached_labels = []
    to_tensor = T.ToTensor()

    for img, label in dataset:
        # Convert tensor → PIL
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)

        original_mode = img.mode  # "L" or "RGB"

        # Compress 
        buf = io.BytesIO()
        if fmt.upper() in ["JPEG", "WEBP"]:
            img.save(buf, format=fmt.upper(), quality=quality)
        else:
            img.save(buf, format=fmt.upper())
        buf.seek(0)

        # Decode while preserving original mode
        comp_img = Image.open(buf).convert(original_mode)

        comp_tensor = to_tensor(comp_img)

        assert comp_tensor.shape[0] == (1 if original_mode == "L" else 3), \
            f"Channel mismatch: expected {original_mode}, got {comp_tensor.shape}"

        cached_imgs.append(comp_tensor)
        cached_labels.append(label)

    cached_imgs = torch.stack(cached_imgs)
    cached_labels = torch.tensor(cached_labels)

    print(f"Cached {len(dataset)} compressed samples.")

    return torch.utils.data.TensorDataset(cached_imgs, cached_labels)

