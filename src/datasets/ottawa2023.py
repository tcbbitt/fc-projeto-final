import numpy as np
import pandas as pd
import random
import os
import torch

class Ottawa2023Dataset(torch.utils.data.Dataset):
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