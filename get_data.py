from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from compress import CompressedDataset, preprocess_and_cache


def get_train_test_data(dataset: str = "fashion",
                         root: str = "data", 
                         batch_size: int = 64, 
                         download: bool = False, 
                         compressor: dict | None = None,):
    """
    Loads MNIST or Fashion-MNIST and returns train + test dataloaders.
    
    dataset: "mnist" or "fashion"
    root: where to store data
    batch_size: dataloader batch size
    """
    
    # transform = transforms.ToTensor()
    if compressor is None:
        transform = transforms.ToTensor()
    else:
        transform = None 

    if dataset.lower() == "fashion":
        trainset = datasets.FashionMNIST(root, train=True, download=download, transform=transform)
        testset  = datasets.FashionMNIST(root, train=False, download=download, transform=transform)
    elif dataset.lower() == "mnist":
        trainset = datasets.MNIST(root, train=True, download=download, transform=transform)
        testset  = datasets.MNIST(root, train=False, download=download, transform=transform)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mnist' or 'fashion'.")
    

    print(f"Loaded dataset {dataset} with {len(trainset)} training samples and {len(testset)} test samples.")
    


    if compressor:
        print(f"Applying compression: {compressor}")
        fmt     = compressor["format"]
        quality = compressor.get("quality", 100)

        # comp_ds_train = CompressedDataset(trainset, fmt, quality, debug=False)
        # comp_ds_test = CompressedDataset(testset, fmt, quality)
        # train_loader = DataLoader(comp_ds_train, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(comp_ds_test, batch_size=batch_size, shuffle=False)

        print(f"Applying compression (cached, one-time): format={fmt}, quality={quality}")

        trainset = preprocess_and_cache(trainset, fmt, quality)
        testset  = preprocess_and_cache(testset, fmt, quality)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
