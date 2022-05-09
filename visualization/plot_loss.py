import json
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
# Everything for creating figures

def plot_losses(train_losses, valid_losses, metric_name, train_params):
    """ Plot losses and save the figure.
    """
    # Get baseline values
    with open(Path("./checkpoint/baseline/metrics.json"), "r") as file:
        baseline_metrics = json.load(file)

    if metric_name == "MSE":
        baseline_valid_loss = baseline_metrics["valid_mse"]
        lstyle = "-"
    elif metric_name == "MAE":
        baseline_valid_loss = baseline_metrics["valid_mae"]
        lstyle = "--"
    elif metric_name == "KL-Divergence":
        baseline_valid_loss = baseline_metrics["valid_kld"]
        lstyle = ":"
    else:
        raise ValueError("Argument 'metric_name' must be one of the following: 'MSE', 'MAE' or 'KL-Divergence'.")

    # plot losses
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")

    # plot baseline loss
    plt.axhline(y = baseline_valid_loss, c='g', linestyle="--", label="DCDI")
    #plt.ylim(0, 0.02)

    # stuff around the plot
    plt.xlabel("epochs")
    plt.legend()
    plt.title("{}: Experiment ID {}, {} TPCNN Units, {} STGCN Units".format(
        metric_name,
        train_params["exp_id"],
        train_params["n_txpcnn"],
        train_params["n_stgcn"],
    ))

    plt.savefig("./checkpoint/exp_{}/losses_{}_{}_{}_{}_{}_{}.pdf".format(
        train_params["exp_id"],
        metric_name,
        train_params["epochs"],
        train_params["batch_size"],
        train_params["lr"],
        train_params["n_txpcnn"],
        train_params["n_stgcn"],
    ))

    if train_params["logger"]:
        plt.show()

    plt.close()

def main():

    exp_id = 20

    metric_names = ["MSE", "MAE", "KL-Divergence"]

    # read in data (loss history in this case)
    with open("./checkpoint/exp_{}/train_metrics.json".format(exp_id), "rb") as file:
        train_metrics = json.load(file)

    with open("./checkpoint/exp_{}/valid_metrics.json".format(exp_id), "rb") as file:
        valid_metrics = json.load(file)

    with open("./checkpoint/exp_{}/args.json".format(exp_id), "rb") as file:
        params = json.load(file)

    for m, metric_name in enumerate(metric_names):
        plot_losses(train_metrics[m], valid_metrics[m], metric_name, params)

if __name__ == "__main__":
    main()
