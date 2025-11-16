
# Configuration settings for image compression experiments
compression_settings = {
    "none":     None,       # Baseline
    # "jpeg_1":   {"format": "JPEG", "quality": 1},
    "jpeg_10":  {"format": "JPEG", "quality": 10},
    "jpeg_50": {"format": "JPEG", "quality": 50},
    "png": {"format": "PNG"},
    "webp_1": {"format": "WEBP", "quality": 1},
    "webp_5": {"format": "WEBP", "quality": 5},
    "webp_20": {"format": "WEBP", "quality": 20},
}

# List of model archs to evaluate
model_list = ["small-MLP"]

# List of datasets to evaluate
datasets_list = [
                "mnist", 
                 "fashion",
                #  "cifar10",
                 ]