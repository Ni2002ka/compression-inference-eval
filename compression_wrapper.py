import io
import sys
import os
import tempfile
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

# Add HiFiC submodule to Python path
HIFIC_PATH = os.path.join(os.path.dirname(__file__), 'hific') 
if HIFIC_PATH not in sys.path:
    sys.path.insert(0, HIFIC_PATH)

# Monkey-patch torch.load to always use CPU for HiFiC
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cpu')
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load

# Import from HiFiC's compress module (now it will find hific/compress.py)
import compress as hific_compress
prepare_model = hific_compress.prepare_model
compress_and_save = hific_compress.compress_and_save
load_and_decompress = hific_compress.load_and_decompress
prepare_dataloader = hific_compress.prepare_dataloader

class HiFiCWrapper:
    """Wrapper for HiFiC compression using existing compress.py functions"""
    def __init__(self, ckpt_path):
        # Create a temporary directory for logs
        self.temp_dir = tempfile.mkdtemp()
        self.model, self.loaded_args = prepare_model(ckpt_path, self.temp_dir)

        # Silence HiFiC logs
        class SilentLogger:
            def info(self, *a, **kw): pass
            def debug(self, *a, **kw): pass
            def warn(self, *a, **kw): pass
            def warning(self, *a, **kw): pass
            def error(self, *a, **kw): pass

        self.model.logger = SilentLogger()

        
    def compress_decompress_tensor(self, img_tensor):
        """
        Compress and decompress a single image tensor in-memory
        Args:
            img_tensor: torch.Tensor of shape (C, H, W) in range [0, 1]
        Returns:
            torch.Tensor of shape (C, H, W) in range [0, 1]
        """
        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save tensor as image
            input_path = os.path.join(tmpdir, "input.png")
            output_compressed_path = os.path.join(tmpdir, "input_compressed.hfc")
            output_recon_path = os.path.join(tmpdir, "reconstruction.png")
            
            # Save the input tensor as PNG
            T.ToPILImage()(img_tensor).save(input_path)
            
            # Prepare dataloader for this single image
            eval_loader = prepare_dataloader(
                self.loaded_args, 
                tmpdir, 
                tmpdir, 
                batch_size=1
            )
            
            # Compress and save
            compress_and_save(self.model, self.loaded_args, eval_loader, tmpdir)
            
            # Load and decompress
            reconstruction = load_and_decompress(
                self.model, 
                output_compressed_path, 
                output_recon_path
            )
            
            # Return as tensor (already in [0, 1] range due to normalize=True in save_image)
            return reconstruction.squeeze(0).cpu()


def preprocess_and_cache(dataset, fmt="JPEG", quality=100, return_ratio=False):
    """
    Preprocess and cache dataset with various compression formats.
    Also optionally returns the average compression ratio
        ratio = compressed_bytes / pixels
    """
    cached_imgs = []
    cached_labels = []
    to_tensor = T.ToTensor()
    
    compression_ratios = []
    
    # Initialize HiFiC wrapper if needed
    hific_wrapper = None
    if fmt.upper() == "HIFIC":
        hific_ckpt = f"models/hific_{quality}.pt"
        print(f"Loading HiFiC model from {hific_ckpt}...")
        hific_wrapper = HiFiCWrapper(hific_ckpt)
        print("HiFiC model loaded successfully.")
    
    for img, label in dataset:

        # --------------------------
        # Compute original size
        # --------------------------
        if isinstance(img, torch.Tensor):
            img_tensor = img
            img_pil = T.ToPILImage()(img)
        else:
            img_pil = img
            img_tensor = to_tensor(img)

        original_mode = img_pil.mode  # "L" or "RGB"
        
        # count pixels to calculate R(D)
        num_pixels = img_tensor.shape[1] * img_tensor.shape[2]
        
        
        # --------------------------
        # Compression
        # --------------------------
        if fmt.upper() == "HIFIC":
            comp_tensor = hific_wrapper.compress_decompress_tensor(img_tensor)
            
            # HiFiC specific: compressed_bytes = latent_size * 1 byte
            comp_bytes = hific_wrapper.last_compressed_size_bytes  # <-- You must expose this in wrapper
            compression_ratios.append(comp_bytes / num_pixels)

        else:
            # Traditional compression (JPEG, PNG, WEBP)
            buf = io.BytesIO()
            if fmt.upper() in ["JPEG", "WEBP"]:
                img_pil.save(buf, format=fmt.upper(), quality=quality)
            else:
                img_pil.save(buf, format=fmt.upper())
            
            comp_bytes = buf.getbuffer().nbytes
            compression_ratios.append(comp_bytes / num_pixels)

            buf.seek(0)
            comp_img = Image.open(buf).convert(original_mode)
            comp_tensor = to_tensor(comp_img)
        
        # Verify shape correctness
        assert comp_tensor.shape[0] == (1 if original_mode == "L" else 3), \
            f"Channel mismatch: expected {original_mode}, got {comp_tensor.shape}"
        
        cached_imgs.append(comp_tensor)
        cached_labels.append(label)
    
    cached_imgs = torch.stack(cached_imgs)
    cached_labels = torch.tensor(cached_labels)
    
    avg_ratio = sum(compression_ratios) / len(compression_ratios) * 8  # bits per pixel
    
    print(f"Cached {len(dataset)} compressed samples using {fmt}. "
          f"Avg compression ratio = {avg_ratio:.4f}")

    if return_ratio:
        return torch.utils.data.TensorDataset(cached_imgs, cached_labels), avg_ratio
    
    return torch.utils.data.TensorDataset(cached_imgs, cached_labels)
