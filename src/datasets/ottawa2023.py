import numpy as np
import pandas as pd
import random
import os
import torch

class Ottawa2023Dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading and processing vibration signals from the Ottawa 2023 dataset.

    Parameters
    ----------
    faulty_type : str, optional
        The type of fault to use as the label. One of: 'inner', 'outer', 'ball', 'cage'. Default is 'inner'.
    normalize : bool, optional
        Whether to normalize the waveform signals. Default is False.
    sample_length : float, optional
        The length of each sample in seconds. Default is 1.
    random_sample : bool, optional
        Whether to randomly sample a window from the signal. Default is False.
    random_gain : bool, optional
        Whether to apply a random gain to the waveform signals. Default is False.

    Attributes
    ----------
    ottawa2023 : pandas.DataFrame
        The loaded Ottawa 2023 dataset.
    faulty_type : str
        The type of fault used as the label.
    normalize : bool
        Whether waveform signals are normalized.
    sample_length : float
        The length of each sample in seconds.
    random_sample : bool
        Whether random sampling is applied to the signal windows.
    random_gain : bool
        Whether random gain is applied to the waveform signals.
    fs : float
        The sampling frequency of the signals.
    samples_per_window : int
        The number of samples in each window.
    windows_per_signal : int
        The number of windows per signal.

    Methods
    -------
    __len__()
        Returns the total number of windows in the dataset.
    __getitem__(idx)
        Retrieves a waveform and its corresponding label based on the given index.

    Notes
    -----
    - The dataset is loaded from a preprocessed pickle file located at
      '/data/bearing_datasets/ottawa/processed/full_dataset.bz2'.
    - The vibration signals are sliced into windows based on the specified sample length.
    - Random sampling and random gain can be applied to the waveform signals for data augmentation.
    - Normalization is performed by subtracting the mean and dividing by the standard deviation of the waveform.
    """
    def __init__(self, faulty_type = 'inner', normalize = False, sample_length = 1, random_sample = False, random_gain = False):
        self.ottawa2023 = pd.read_pickle('/data/bearing_datasets/ottawa/processed/full_dataset.bz2')
        self.faulty_type = faulty_type
        self.normalize = normalize
        self.sample_length = sample_length
        self.random_sample = random_sample
        self.random_gain = random_gain
        self.fs = self.ottawa2023['fs'].iloc[0]
        self.samples_per_window = int(self.fs * self.sample_length)
        self.windows_per_signal = int(10/self.sample_length)
        
    def __len__(self):
        return len(self.ottawa2023)*self.windows_per_signal

    def __getitem__(self, idx):
        signal_idx = idx // self.windows_per_signal
        window_idx = idx % self.windows_per_signal
        row = self.ottawa2023.iloc[signal_idx]
        signal = row['vibration']  # numpy array
        label = row[self.faulty_type]
        #
        if self.random_sample:
                # Random start index within [0, max_start] to get 1s slice
                max_start = len(signal) - self.samples_per_window
                start = random.randint(0, max_start)
        else:
                start = window_idx * self.samples_per_window

        end = start + self.samples_per_window
        waveform = signal[start:end]
        if self.random_gain:
            gain = random.uniform(0.5, 1.5)
            waveform = waveform * gain

        if self.normalize:
            waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-8) 
            
        waveform = torch.tensor(waveform, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return waveform, label