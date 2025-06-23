import torch
from sklearn.metrics import roc_auc_score
import wandb

def train(dataloader, model, loss_fn, optimizer, device, epoch):
    """
    Runs one epoch of training and logs average loss and AUROC to wandb.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the training dataset.
    model : torch.nn.Module
        Neural network model to train.
    loss_fn : callable
        Loss function (e.g., BCEWithLogitsLoss).
    optimizer : torch.optim.Optimizer
        Optimizer (e.g., Adam).
    device : torch.device
        Device to move data and model to (e.g., 'cuda' or 'cpu').
    epoch : int
        Current epoch number (for logging).

    Returns
    -------
    float
        Average loss over the epoch.
    float
        AUROC over the epoch.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).unsqueeze(1)  # (B,) -> (B,1)

        optimizer.zero_grad()

        outputs = model(X)  # raw logits output shape: (B,1)

        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Collect preds and targets for AUROC
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        targets = y.detach().cpu().numpy()

        all_preds.extend(probs.flatten())
        all_targets.extend(targets.flatten())

    avg_loss = total_loss / len(dataloader)
    
    # Compute AUROC
    auroc = roc_auc_score(all_targets, all_preds)


    # Log metrics to wandb
    wandb.log({
        "train_loss": avg_loss,
        "train_auroc": auroc,
        "epoch": epoch
    })

    return avg_loss, auroc
