import csv
import time
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from utils import timeKLDivLoss
from model import CausalSTGCN
from plotting import plot_losses
from data_preprocessing import DataProcesser


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    """
    """
    # create a data preprocesser, say which data it should use
    path_data = ("dataset/data{}/".format(args.dataset_num)) #("../exp_dataset/data14/")
    processer = DataProcesser(path_data)
    # read out data params
    with open("dataset/data{}/data_params.json".format(args.dataset_num), "r") as file:
        data_params = json.load(file)

    # get torch dataloaders
    train_loader, valid_loader = processer.get_dataloaders(
        batch_size=args.batch_size,
        train_valid_split=0.8
    )

    # read this from args
    train_params = {
        "epochs": args.epochs,#5,
        "batch_size": args.batch_size, #32,
        "lr": args.lr,#0.001, #0.001
        "optimizer": args.optimizer, #"Adam",
        "lr_scheduler": args.lr_scheduler, #False,
        "exp_id": args.exp_id, #6,
        "logger": args.logger, #False,
        "spatial_kernel_size": args.kernel_size, #3,
        "n_stgcn": args.stgcn,#10, # 0 stgcn and 10 txpcnn is good..., nut 2 - 0 as well!
        "n_txpcnn": args.tpcnn, #1,
    }
    # model
    model = CausalSTGCN(
        n_stgcn=train_params["n_stgcn"],
        n_txpcnn=train_params["n_txpcnn"],
        input_feat=data_params["num_features"],
        output_feat=1,
        seq_len=data_params["timewindow"]+1,
        pred_seq_len=data_params["timewindow"]+1,
        kernel_size=train_params["spatial_kernel_size"]
    )
    model.to(device)

    # loss function
    loss_func = nn.MSELoss()

    # optimizer
    if train_params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"])
    elif train_params["optimizer"] == "SDG":
        optimizer = torch.optim.SDG(model.parameters(), lr=train_params["lr"])
    if train_params["lr_scheduler"]:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_params["lr"],
            gamma=0.2
        )

    # metrics and storage
    best_valid_loss = 100 # just an init value
    checkpoint_dir = Path("./checkpoint/exp_{}/".format(train_params["exp_id"]))
    if not checkpoint_dir.is_dir():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "args.json", "w") as fp:
        data = json.dump(train_params, fp,
            indent = 4,
            sort_keys = True,
            separators = (',', ': '),
            ensure_ascii = False,
        )
    train_losses = []
    train_maes = []
    train_klds = []
    valid_losses = []
    valid_maes = []
    valid_klds = []
    # DO NOT use as metric - only to estimate running time
    # for metric use: https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    training_time = []

    # get logger
    logger = train_params["logger"]
    # train epochs
    print("Start Training ...")
    for epoch in range(train_params["epochs"]):
        # measure running time
        start = time.time()

        # train the model
        batch_losses, batch_maes, batch_klds = train_epoch(model, train_loader, loss_func, optimizer, logger=logger)
        #print("\nEpoch {:3} Train Loss: {:7.6f}".format(epoch, np.mean(batch_losses)))

        # validate the model
        valid_loss, valid_mae, valid_kld = valid_epoch(model, valid_loader, loss_func)
        #print("          Valid Loss: {:7.6f}\n".format(valid_loss))

        # lr schedule
        if train_params["lr_scheduler"]:
            scheduler.step()

        # save metrics
        train_losses.append(np.mean(batch_losses))
        train_maes.append(np.mean(batch_maes))
        train_klds.append(np.mean(batch_klds))
        valid_losses.append(valid_loss)
        valid_maes.append(valid_mae)
        valid_klds.append(valid_kld)

        # save model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_dir / "val_best.pth")

        # measure running time
        end = time.time()
        train_time = end - start
        training_time.append(train_time)

        print("\nEpoch {:3} | Train Loss: {:7.6f} | Valid Loss: {:7.6f} \\\\ Elapsed Time: {:5.4}".format(epoch, np.mean(batch_losses), valid_loss, train_time))

    # store all relevant information in csv
    if args.store_csv:
        data = [
            args.exp_id, args.epochs, args.batch_size, args.lr, args.lr_scheduler,
            args.optimizer, args.kernel_size, args.stgcn, args.tpcnn,
            data_params["exp_id"], data_params["n"], data_params["num_gridcells"],
            data_params["prob"], data_params["neighborhood"],
            data_params["timewindow"], data_params["eta"],
            data_params["num_features"], np.mean(training_time),
            np.min(train_losses), np.min(train_maes), np.min(train_klds),
            np.min(valid_losses), np.min(valid_maes), np.min(valid_klds),
        ]
        with open(Path("./tuning/results.csv"), 'a', encoding="UTF8") as file:
            writer = csv.writer(file)
            writer.writerow(data)

    # plot
    plot_losses(train_losses, valid_losses, "MSE", train_params)

    # store losses
    with open(checkpoint_dir / "train_metrics.json", "w") as fp:
        json.dump([train_losses, train_maes, train_klds, training_time], fp)
    with open(checkpoint_dir / "valid_metrics.json", "w") as fp:
        json.dump([valid_losses, valid_maes, valid_klds, training_time], fp)

    # finished
    print("Training finished!")


def train_epoch(model, train_loader, loss_func, optimizer, logger=True):
    """ Trains a given model for one epoch
    Params:
        model ():
        epoch_idx (int): number of epoch we are in
        train_loader (torch.DataLoader): training data
        optimizer (torch.optim.Optimizer): a torch optimizer
        loss_func (torch.nn.LossFunction): a torch loss function
        logger (bool): if True, batch losses are printed
    Returns:
        (float, list <float>): last loss and list of losses
    """
    # define additional metrics
    mae_func = nn.L1Loss()
    kld_func = timeKLDivLoss

    # store loss and metrics
    batch_losses = []
    running_loss = 0.0
    maes = []
    klds = []

    for i, data in enumerate(train_loader):
        # data: (x, adj, baseline), y
        x = data[0][0].to(device)
        y = data[1].to(device)
        # we don't need a batch of adj, since its always the same
        adj = data[0][1][0,].to(device)

        # optimizer
        optimizer.zero_grad()

        # make prediction for this batch
        outputs, _ = model(x, adj)

        # compute loss and gradients
        loss = loss_func(outputs, y)
        loss.backward()

        # If needed: clip gradients
        # if clip_grad is not None:
        #     nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # update learning weights
        optimizer.step()

        # other metrics
        maes.append(mae_func(outputs, y).item())
        klds.append(kld_func(outputs, y).item())

        # print loss information
        running_loss += loss.item()
        batch_losses.append(loss.item())
        if logger:
            print("    Batch {:3} Loss: {:10.9f}".format(i, loss.item()))

    #epoch_loss = running_loss / len(train_loader)
    return batch_losses, maes, klds


def valid_epoch(model, valid_loader, loss_func):
    """
    """
    # define additional metrics
    mae_func = nn.L1Loss()
    kld_func = timeKLDivLoss

    running_loss = 0.0
    running_mae = 0.0
    running_kld = 0.0

    for i, data in enumerate(valid_loader):
        x = data[0][0].to(device)
        y = data[1].to(device)
        # we don't need a batch of adj, since its always the same
        adj = data[0][1][0,].to(device)

        # make prediction for this batch
        outputs, _ = model(x, adj)

        # compute loss and metrics
        loss = loss_func(outputs, y)
        mae = mae_func(outputs, y)
        kld = kld_func(outputs, y)

        running_loss += loss.item()
        running_mae += mae.item()
        running_kld += kld.item()

    valid_loss = running_loss / len(valid_loader)
    valid_mae = running_mae / len(valid_loader)
    valid_kld = running_kld / len(valid_loader)

    return valid_loss, valid_mae, valid_kld

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train STGCN model for causal graphs.")
    parser.add_argument("--exp_id", type=int,
                        help="Unique identifier for this experiment")
    # TODO maybe use datapath instead
    parser.add_argument("--dataset_num", type=int, default=4,
                        help="Which dataset should be used")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--lr_scheduler", action="store_true",
                        help="Use learning rate scheduler")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer should be used")
    parser.add_argument("--logger", action="store_true",
                        help="Logging more information")
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="Spatial kernel size")
    parser.add_argument("--stgcn", type=int, default=10,
                        help="How many STGCN Blocks should be used")
    parser.add_argument("--tpcnn", type=int, default=2,
                        help="How many TPCNN Blocks should be used")
    parser.add_argument("--store_csv", action="store_true",
                        help="Whether the results should be written to a shared csv file.")

    args = parser.parse_args()
    main(args)
