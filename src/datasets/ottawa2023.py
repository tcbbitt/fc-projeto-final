import numpy as np
import pandas as pd
import random
import os
import torch
from src.datasets.transforms import ToTensor, FFT, Normalize, RandomGain

class Ottawa2023Dataset(torch.utils.data.Dataset):
    def __init__(self, data_df, faulty_type='inner', sample_length=1.0, transform=None, random_sample=False, seed=42):
        self.ottawa2023 = data_df.reset_index(drop=True)
        self.faulty_type = faulty_type
        self.sample_length = sample_length
        self.transform = transform
        self.random_sample = random_sample
        self.rng = random.Random(seed)
        self.fs = self.ottawa2023['fs'].iloc[0]
        self.samples_per_window = int(self.fs * self.sample_length)
        self.windows_per_signal = int(10 / self.sample_length)

    def __len__(self):
        return len(self.ottawa2023) * self.windows_per_signal

    def __getitem__(self, idx):
        signal_idx = idx // self.windows_per_signal
        window_idx = idx % self.windows_per_signal
        row = self.ottawa2023.iloc[signal_idx]
        signal = row['vibration']
        label = row[self.faulty_type]

        if self.random_sample:
            max_start = len(signal) - self.samples_per_window
            start = self.rng.randint(0, max_start)
        else:
            start = window_idx * self.samples_per_window

        waveform = signal[start:start + self.samples_per_window]
        sample = (waveform, label)

        if self.transform:
            sample = self.transform(sample)

        return sample
