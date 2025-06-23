from torch.utils.data import Dataset

class multiplydataset(Dataset):
    """
    Dataset wrapper that repeats the underlying dataset multiple times.

    This class creates a virtual dataset that cycles through the original dataset
    multiple times without duplicating data in memory. Useful for effectively increasing
    the size of small datasets during training.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The original dataset to be repeated.
    repeats : int, optional
        Number of times to repeat the dataset. Default is 1 (no repetition).

    Methods
    -------
    __len__()
        Returns the length of the repeated dataset (original length multiplied by repeats).
    __getitem__(idx)
        Retrieves the sample from the original dataset corresponding to the
        repeated index `idx`, cycling through the original dataset.

    Notes
    -----
    The indexing wraps around using modulo operation to cycle through the original dataset.
    """

    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = repeats
        self.original_length = len(dataset)

    def __len__(self):
        return self.original_length * self.repeats

    def __getitem__(self, idx):
        # Cycle idx to original dataset length
        original_idx = idx % self.original_length
        return self.dataset[original_idx]
