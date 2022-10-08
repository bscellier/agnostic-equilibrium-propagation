import torch
import torchvision


class IndexedDataset(torch.utils.data.Dataset):
    """
    Class for an indexed dataset.

    Each item of an indexed dataset consists of an image, its label, and its index in the dataset.
    In contrast, the standard Dataset class does not return the index of the image
    """

    def __init__(self, dataset):
        self._dataset = dataset
        
    def __getitem__(self, index):
        data, target = self._dataset[index]
        
        return data, target, index

    def __len__(self):
        return len(self._dataset)


def load_mnist():
    """Loads the MNIST dataset (training and test sets).

    Returns:
        tuple of datasets: the training dataset and the test dataset
    """

    training_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    return training_data, test_data


def load_fashion_mnist():
    """Loads the FashionMNIST dataset (training and test sets).

    Returns:
        tuple of datasets: the training dataset and the test dataset
    """

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.,), std=(1,))
        ])

    training_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.,), std=(1,))
        ])

    test_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
    )

    return training_data, test_data


def load_dataset(dataset):
    """Loads the dataset (training and test sets).

    Args:
        dataset (str): The dataset used for training. Either 'MNIST' or 'FashionMNIST'

    Returns:
        tuple of datasets: the training dataset and the test dataset
    """

    if dataset == 'MNIST':
        training_data, test_data = load_mnist()

    elif dataset == 'FashionMNIST':
        training_data, test_data = load_fashion_mnist()

    else:
        raise ValueError("expected 'MNIST' or 'FashionMNIST' but got {}".format(dataset))

    return training_data, test_data


def load_dataloaders(dataset, batch_size):
    """Builds data loaders (training and test loaders).

    Args:
        dataset (str): The dataset used for training. Either 'MNIST' or 'FashionMNIST'
        batch_size (int): size of the mini-batch

    Returns:
        tuple of dataloaders: the training loader and the test loader
    """

    training_data, test_data = load_dataset(dataset)
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_data = IndexedDataset(test_data)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

    return training_loader, test_loader
