from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from compression_wrapper import preprocess_and_cache


def get_train_test_data(dataset: str = "fashion",
                         root: str = "data", 
                         batch_size: int = 64,
                         train: bool = True,
                         test: bool = True, 
                         download: bool = False, 
                         compressor: dict | None = None,
                         get_ratio:  bool = False):
    """
    Loads MNIST or Fashion-MNIST and returns train + test dataloaders.
    
    dataset: "mnist" or "fashion"
    root: where to store data
    batch_size: dataloader batch size
    train: whether to load training data
    test: whether to load test data
    download: whether to download data if not present
    compressor: dict with compression settings, e.g. {"format": "JPEG", "quality": 10}
    get_ratio: whether to return average compression ratio
    """
    
    train_loader = None
    test_loader  = None

    # transform = transforms.ToTensor()
    if compressor is None:
        transform = transforms.ToTensor()
    else:
        transform = None 

    if dataset.lower() == "fashion":
        if train:
            trainset = datasets.FashionMNIST(root, train=True, download=download, transform=transform)
        if test:
            testset  = datasets.FashionMNIST(root, train=False, download=download, transform=transform)
    elif dataset.lower() == "mnist":
        if train:
            trainset = datasets.MNIST(root, train=True, download=download, transform=transform)
        if test:
            testset  = datasets.MNIST(root, train=False, download=download, transform=transform)
    elif dataset == "cifar10":
        if train:
            trainset = datasets.CIFAR10(root, train=True, download=download, transform=transform)
        if test:
            testset  = datasets.CIFAR10(root, train=False, download=download, transform=transform)
    elif dataset == "stl10":
        if train:
            trainset = datasets.STL10(root, split='train', download=download, transform=transform)
        if test:
            testset  = datasets.STL10(root, split='test', download=download, transform=transform)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mnist', 'fashion', or 'cifar10'.")
    

    print(f"Loaded dataset {dataset}.")
    

    ratio = 24.0 if dataset in ["cifar10", "stl10"] else 8.0  # Default ratio (no compression)
    if compressor:
        print(f"Applying compression: {compressor}")
        fmt     = compressor["format"]
        quality = compressor.get("quality", 100)

        print(f"Applying compression (cached, one-time): format={fmt}, quality={quality}")

        if train:
            trainset, ratio = preprocess_and_cache(trainset, fmt, quality, return_ratio=True)
        if test:
            testset, ratio  = preprocess_and_cache(testset, fmt, quality, return_ratio=True)

    if train:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    if test:
        test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False)

    if get_ratio:
        return train_loader, test_loader, ratio
    
    return train_loader, test_loader
