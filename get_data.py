import torchvision.datasets as datasets
from torchvision import transforms

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
