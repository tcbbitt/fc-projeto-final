import autorootcwd
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from src.datasets.splits import get_ottawa2023_splits
from src.datasets.ottawa2023 import Ottawa2023Dataset
from src.datasets.dataset_wrapper import multiplydataset
from src.datasets.transforms import ToTensor, Normalize, FFT, RandomGain
from src.datasets.dataloaders import get_dataloaders
from src.models.dcnn import Net
from src.trainer.train import train
from src.trainer.test import test
import wandb
def main():
    
    # --- Configs ---
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 20
    FAULT_TYPE = "cage"
    SAMPLE_LENGTH = 1.0 #in seconds
    REPEATS = 10
    LEARNING_RATE = 1e-3
    AUGMENTATION = 'normalized-fft'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DICT OF AUGMENTATIONS
    augmentation = { 'fft-RG': [FFT(), RandomGain(seed=SEED), ToTensor()],
                        'fft': [FFT(), ToTensor()],
                        'normalized-fft' : [Normalize(), FFT(), ToTensor()],
                        'normalized-time': [Normalize(), ToTensor()],
                        'time-RG': [RandomGain(seed=SEED), ToTensor()],
                        'time': [ToTensor()] }     
    # --- Define transforms ---
    train_transform = transforms.Compose(augmentation[AUGMENTATION])
    val_test_transform = transforms.Compose(augmentation[AUGMENTATION])     
    wandb.login()
    wandb.init(project="ottawa2023-fault-detection", 
           name=f"DCNN08_{FAULT_TYPE}_{AUGMENTATION}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{EPOCHS}epochs",
           config={
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "model": "DCNN08",
    "fault_type": FAULT_TYPE,
    "augmentation": AUGMENTATION,
    })
    #load data
    full_df = pd.read_pickle('/data/bearing_datasets/ottawa/processed/full_dataset.bz2')
    #split
    train_df, val_df, test_df = get_ottawa2023_splits(full_df)
    # --- Create datasets ---
    train_dataset = Ottawa2023Dataset(train_df, faulty_type=FAULT_TYPE, sample_length=SAMPLE_LENGTH, transform=train_transform, random_sample=True)
    val_dataset   = Ottawa2023Dataset(val_df, faulty_type=FAULT_TYPE, sample_length=SAMPLE_LENGTH, transform=val_test_transform, random_sample=False)
    test_dataset  = Ottawa2023Dataset(test_df, faulty_type=FAULT_TYPE, sample_length=SAMPLE_LENGTH, transform=val_test_transform, random_sample=False)
    # --- Multiply training set ---
    train_dataset = multiplydataset(train_dataset, repeats=REPEATS)
    # --- Create dataloaders ---
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    # --- check device ---
    print(f"Using device: {DEVICE}")
    # --- Model, Loss, Optimizer ---
    model = Net("DCNN08", in_channels=1, n_class=1).to(DEVICE)
    loss_BCE = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # --- Train and Test Loop ---
    for epoch in range(EPOCHS):
        train_loss, auroc = train(dataloader = train_loader, model = model, loss_fn = loss_BCE, optimizer = optimizer, device = DEVICE, epoch=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, AUROC: {auroc:.4f}")
        test(dataloader = val_loader, model = model, loss_fn = loss_BCE, device = DEVICE, epoch = epoch, val_or_test='val')
        test(dataloader = test_loader, model = model, loss_fn = loss_BCE, device = DEVICE, epoch = epoch, val_or_test='test')

if __name__ == "__main__":
    main()
