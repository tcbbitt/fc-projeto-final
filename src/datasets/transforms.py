import numpy as np
import torch

class ToTensor:
    """Convert numpy arrays in sample to PyTorch tensors."""
    def __call__(self, sample):
        waveform, label = sample
        waveform = torch.tensor(waveform, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return waveform, label

class FFT:
    """Apply FFT magnitude transform to waveform."""

    def __call__(self, sample):
        waveform, label = sample
        fft_result = np.fft.fft(waveform)
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        return magnitude, label

class Normalize:
    """Normalize waveform to zero mean and unit std."""
    def __call__(self, sample):
        waveform, label = sample
        mean = np.mean(waveform)
        std = np.std(waveform) + 1e-8
        waveform = (waveform - mean) / std
        return waveform, label

class RandomGain:
    """Apply random gain scaling to waveform."""
    def __init__(self, min_gain=0.5, max_gain=2.5, seed=None):
        import random
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.rng = random.Random(seed)

    def __call__(self, sample):
        waveform, label = sample
        gain = self.rng.uniform(self.min_gain, self.max_gain)
        waveform = waveform * gain
        return waveform, label
