import json
import torch
import torch.nn as nn

from pathlib import Path

from utils import timeKLDivLoss
from data_preprocessing import DataProcesser

def baseline(train_loader, valid_loader, loss_func):
    """
    Params:
        train_loader (torch.DataLoader): training data
        valid_loader (torch.DataLoader): validation data
    Returns:
        (float, float): train_loss, valid_loss
    """
    train_loss = 0
    valid_loss = 0

    for batch in train_loader:
        (_, _, baseline_pred), y = batch
        train_loss += loss_func(baseline_pred, y)

    for batch in valid_loader:
        (_, _, baseline_pred), y = batch
        valid_loss += loss_func(baseline_pred, y)

    train_loss = train_loss / len(train_loader)
    valid_loss = valid_loss / len(valid_loader)

    return train_loss, valid_loss

def run_baseline(data_dir, split, batch_size, loss_funcs):
    """
    Params:
        data_dir (str): Path to dataset
        split (float): train-valid split for baseline
        batch_size (int): must be the same like for the other models
        loss_funcs (dict <str:nn.LossFunction>): Loss functions used for
            the other models as well. Dictionary with names as keys, and
            pytorch loss functions as values
    """
    # create processer
    processer = DataProcesser(data_dir)

    # get input and ground truth data
    train_loader, valid_loader = processer.get_dataloaders(
        batch_size=batch_size,
        train_valid_split=split,
    )

    # get metrics from baseline
    metrics = {}
    for loss_name, loss_func in loss_funcs.items():
        train_loss, valid_loss = baseline(train_loader, valid_loader, loss_func)
        metrics["train_{}".format(loss_name)] = train_loss.item()
        metrics["valid_{}".format(loss_name)] = valid_loss.item()

    # save results
    with open(Path("checkpoint/baseline/metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent = 4, sort_keys = True,
            separators = (',', ': '), ensure_ascii = False,
        )

    print("Baseline \nTrain Loss: {:8.7f}\nValid Loss: {:8.7f}".format(metrics["train_mse"], metrics["valid_mse"]))

if __name__ == "__main__":
    # relevant data
    data_dir = ("../dataset/data14/")
    split = 0.8
    batch_size = 32
    kld = timeKLDivLoss
    loss_funcs = {
        "kld": kld,
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
    }

    run_baseline(data_dir, split, batch_size, loss_funcs)
