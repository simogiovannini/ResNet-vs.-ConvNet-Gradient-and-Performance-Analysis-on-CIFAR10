import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T


def load_cifar10():
    train_data = datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.Resize(size=(128, 128), antialias=True)
        ]),
        target_transform=None
    )

    splits = torch.utils.data.random_split(train_data, [40000, 10000])

    train_data = splits[0]
    val_data = splits[1]

    test_data = datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.Resize(size=(128, 128), antialias=True)
        ]),
    )

    return train_data, val_data, test_data


def create_dataloaders(train_data, val_data, test_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
