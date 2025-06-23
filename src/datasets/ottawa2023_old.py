import numpy as np
import pandas as pd
import random
import os
import torch

class Ottawa2023Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading and processing vibration signals from the Ottawa 2023 bearing fault dataset.

    Parameters
    ----------
    faulty_type : str, optional
        Fault type to use as label. One of: 'inner', 'outer', 'ball', 'cage'. Default is 'inner'.
    normalize : bool, optional
        Whether to normalize waveform signals (zero mean, unit variance). Default is False.
    sample_length : float, optional
        Length of each data window in seconds. Default is 1.
    fft : bool, optional
        Whether to apply FFT magnitude transform to each waveform window. Default is False.
    random_sample : bool, optional
        Whether to randomly sample windows within signals (instead of fixed slicing). Default is False.
    random_gain : bool, optional
        Whether to apply random gain augmentation to waveform signals. Default is False.
    seed : int, optional
        Seed for reproducible random sampling and gain. Default is 42.

    Attributes
    ----------
    ottawa2023 : pandas.DataFrame
        Loaded dataset containing vibration signals and labels.
    faulty_type : str
        Fault type used for labels.
    normalize : bool
        Flag for normalization.
    sample_length : float
        Window length in seconds.
    fft : bool
        Flag for FFT magnitude transform.
    random_sample : bool
        Flag for random window sampling.
    random_gain : bool
        Flag for random gain augmentation.
    seed : int
        Seed for reproducibility.
    rng : random.Random
        Local RNG instance for deterministic behavior.
    fs : float
        Sampling frequency of vibration signals.
    samples_per_window : int
        Number of samples per data window.
    windows_per_signal : int
        Number of windows per 10-second signal.

    Methods
    -------
    __len__()
        Returns total number of data windows in the dataset.
    __getitem__(idx)
        Returns a tuple (waveform, label) for the window at index `idx`.

    Notes
    -----
    - Dataset is loaded from '/data/bearing_datasets/ottawa/processed/full_dataset.bz2'.
    - Each signal is assumed to be 10 seconds long.
    - Supports optional data augmentation via random sampling and gain.
    - Normalization and FFT are applied after augmentation, if enabled.
    """

    def __init__(self, faulty_type = 'inner', normalize = False, sample_length = 1, fft = False, random_sample = False, random_gain = False, seed = 42):
        self.ottawa2023 = pd.read_pickle('/data/bearing_datasets/ottawa/processed/full_dataset.bz2')
        self.faulty_type = faulty_type
        self.normalize = normalize
        self.sample_length = sample_length
        self.random_sample = random_sample
        self.random_gain = random_gain
        self.fft = fft
        self.seed = seed
        self.rng = random.Random(seed)  # 
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
                start = self.rng.randint(0, max_start)
        else:
                start = window_idx * self.samples_per_window

        end = start + self.samples_per_window
        waveform = signal[start:end]
        
        if self.random_gain:
            gain = self.rng.uniform(0.5, 1.5)
            waveform = waveform * gain
            # waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-8)  # zero mean, unit std
            # waveform = waveform * target_std + 1  # final mean = 1, std = target_std
            
        if self.fft:
            waveform = np.fft.fft(waveform)
            waveform = np.abs(waveform[0:self.samples_per_window//2]) 

        if self.normalize:
            waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-8) 
            
        waveform = torch.tensor(waveform, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return waveform, label