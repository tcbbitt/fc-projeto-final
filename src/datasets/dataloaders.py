from torch.utils.data import DataLoader

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16, num_workers=4):
    """
    get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16, num_workers=4)

    Creates PyTorch DataLoader objects for training, validation, and testing datasets.

    Parameters
    ----------
    train_dataset : torch.utils.data.Dataset
        The dataset to be used for training.
    val_dataset : torch.utils.data.Dataset
        The dataset to be used for validation.
    test_dataset : torch.utils.data.Dataset
        The dataset to be used for testing.
    batch_size : int, optional
        The number of samples per batch to load (default is 16).
    num_workers : int, optional
        The number of subprocesses to use for data loading (default is 4).

    Returns
    -------
    tuple
        A tuple containing three DataLoader objects:
        - train_loader : DataLoader for the training dataset.
        - val_loader : DataLoader for the validation dataset.
        - test_loader : DataLoader for the testing dataset.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader
