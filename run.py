#############################
#        MAIN PIPELINE      #
#############################

def run_pipeline(models, dataset_path, compression_settings, batch_size=64, device="cpu"):
    results = {}

    # Baseline dataset
    transform = T.Compose([T.ToTensor()])
    orig = datasets.ImageFolder(dataset_path, transform=transform)
    orig_loader = DataLoader(orig, batch_size=batch_size, shuffle=False)

    # Pre-generate compressed loaders
    compressed_loaders = {}
    for name, cfg in compression_settings.items():
        comp_ds = apply_compression(orig, cfg)
        compressed_loaders[name] = DataLoader(comp_ds, batch_size=batch_size, shuffle=False)

    # Evaluation
    for model_name, model in models.items():
        model = model.to(device)
        results[model_name] = {}

        # Baseline
        acc, latency = evaluate(model, orig_loader, device)
        results[model_name]["original"] = {"accuracy": acc, "latency": latency}

        # Compressed
        for comp_name, loader in compressed_loaders.items():
            acc, latency = evaluate(model, loader, device)
            results[model_name][comp_name] = {"accuracy": acc, "latency": latency}

    return results


import torchvision.models as models

models_to_test = {
    "resnet18": models.resnet18(weights="IMAGENET1K_V1"),
    "mobilenet_v3": models.mobilenet_v3_large(weights="IMAGENET1K_V1"),
}



compression_settings = {
    "jpeg_10": {"format": "JPEG", "quality": 10},
    "jpeg_50": {"format": "JPEG", "quality": 50},
    "webp_20": {"format": "WEBP", "quality": 20},
    "png": {"format": "PNG"},
}


results = run_pipeline(
    models=models_to_test,
    dataset_path="/path/to/testdata",
    compression_settings=compression_settings,
)

print(results)
