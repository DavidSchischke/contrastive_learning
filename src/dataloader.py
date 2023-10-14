from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor

def load_emnist(batch_size: int = 512, split: str = "balanced", shuffle: bool = True, root: str = "data") -> [DataLoader, DataLoader]: 
    """Load MNIST Dataset from memory or download it if it is not found

    Args:
        batch_size (int): Batch Size for DataLoader
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        root (str, optional): Path to store the data. Defaults to "data".

    Returns:
        tuple[MNIST, MNIST]: (train_data, test_data)
    """

    try:
        train = EMNIST(
            root=root,
            split = split,
            train=True,
            download=False,
            transform = ToTensor()
        )

        test = EMNIST(
            root=root,
            split = split,
            train=False,
            download=False,
            transform = ToTensor()
        )
    except RuntimeError:
        train = EMNIST(
            root=root,
            split = split,
            train=True,
            download=True,
        )

        test = EMNIST(
            root=root,
            split = split,
            train=False,
            download=True,
        )

    train_data = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_data = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    return train_data, test_data

if __name__ == "__main__": 
    load_emnist()