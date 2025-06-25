import torch
from sklearn.metrics import roc_auc_score
import wandb

def test(dataloader, model, loss_fn, device, epoch, val_or_test='val'):
    """
    Evaluate model on validation or test set and log to wandb.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Validation or test dataloader.
    model : torch.nn.Module
        Trained model.
    loss_fn : callable
        Loss function used.
    device : torch.device
        Device to run on ('cuda' or 'cpu').
    epoch : int
        Current epoch number.
    val_or_test : str
        Either 'val' or 'test' to log appropriately.

    Returns
    -------
    avg_loss : float
    auroc : float
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = len(dataloader)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            all_preds.extend(torch.sigmoid(pred).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = test_loss / num_batches
    auroc = roc_auc_score(all_labels, all_preds)
    # Convert to float for logging
    print(f"[{val_or_test}] Epoch {epoch} - Loss: {avg_loss:.4f}, {val_or_test}_AUROC: {auroc:.4f}")

    wandb.log({
        f"{val_or_test}_loss": avg_loss,
        f"{val_or_test}_auroc": auroc,
        "epoch": epoch
    })

    return avg_loss, auroc
