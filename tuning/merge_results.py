import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
# TODO make this more efficient
def merge_results():
    all_results = []
    header = None

    # get column names
    with open(Path("header.csv"), 'r') as file:
        csv_reader = csv.reader(file)
        results = [row for row in csv_reader]
        header = results[0]

    # Run over checkpoint dir
    for exp_id in range(72):
        # get each results.csv file
        with open(Path("../checkpoint/exp_{}/results.csv".format(exp_id)), 'r') as file:
            csv_reader = csv.reader(file)
            results = [row for row in csv_reader]
            all_results.append(results[0])

    # create pandas frame from that
    results_df = pd.DataFrame(all_results, columns=header)

    # put all results into all_results.csv
    results_df.to_csv("all_results.csv")

    return results_df

def print_result_stats(results):
    """
    """
    # mask away unnnecessary information
    results = results.drop(columns=["epochs", "batch_size", "optimizer", "data_id", "n", "num_gridcells", "prob", "neighborhood", "timewindow", "eta", "num_features"], axis=1)

    # filter out results with stgcn=0 and tpcnn=0
    results = results[~((results["tpcnn"] == "0") & (results["stgcn"] == "0"))]

    # print best model with lowest mse, mae and kld
    min_mse = results.valid_mse.min()
    min_mae = results.valid_mae.min()
    min_kld = results.valid_kld.min()
    print("Best model MSE: \n", results[results.valid_mse == min_mse])
    print("Best model MAE: \n", results[results.valid_mae == min_mae])
    print("Best model KLD: \n", results[results.valid_kld == min_kld])

    # get worst model
    max_mse = results.valid_mse.max()
    max_mae = results.valid_mae.max()
    max_kld = results.valid_kld.max()
    print("Worst model MSE: \n", results[results.valid_mse == max_mse])
    print("Worst model MAE: \n", results[results.valid_mae == max_mae])
    print("Worst model KLD: \n", results[results.valid_kld == max_kld])

def plot_heatmap(results):
    """
    """
    fig, axs = plt.subplots(1, 3, sharex = True, sharey = True, figsize=(15, 5))

    vmax = {"mse": 0.01, "mae": 0.1, "kld": 4}
    vmin = {"mse": 0, "mae": 0, "kld": 3}

    for ax, metric in zip(axs, ["mse", "mae", "kld"]):
        data = np.zeros(shape=(3,3))
        # repeat three times for different metrics
        res = results[["tpcnn", "stgcn", "valid_{}".format(metric)]]
        for idx_t, t in enumerate(["0", "2", "10"]):
            for idx_g, g in enumerate(["0", "2", "10"]):
                # calculate average performance
                perf = (res[(res["tpcnn"] == t) & (res["stgcn"] == g)])["valid_{}".format(metric)]
                perf = [float(p) for p in perf]
                data[idx_t][idx_g] = np.mean(perf)

        data = np.flip(data, axis=0)
        sns.heatmap(data, ax=ax, linewidths=.5, cmap="YlGnBu", vmin=vmin[metric], vmax=vmax[metric])
        ax.set_xticks([0.5, 1.5, 2.5], labels=["0", "2", "10"])
        ax.set_yticks([0.5, 1.5, 2.5], labels=["10", "2", "0"])

    fig.text(0.5, 0.04, "STGCN Units", ha='center')
    fig.text(0.04, 0.5, "TPCNN Units", va='center', rotation='vertical')
    #plt.show()

    plt.savefig("../visualization/heatmap.pdf")

    # get the relevant data
    # data = []
    # for t in range(3):
    #     for g in range(3):

def get_table_values(results):
    """
    """
    # baseline is not here
    stgcn = results[(results["stgcn"] == "2") & (results["tpcnn"] == "0")]
    pcnn = results[(results["stgcn"] == "0") & (results["tpcnn"] == "2")]
    both = results[(results["stgcn"] == "2") & (results["tpcnn"] == "2")]

    for data, name in zip([stgcn, pcnn, both], ["STGCN", "PCNN", "Both"]):
        print("{} values:".format(name))
        for metric in ["valid_mse", "valid_mae", "valid_kld"]:
            perf = [float(p) for p in data[metric]]
            perf_mean = np.mean(perf)
            print("\t {}: {:6.5f}".format(metric, perf_mean))


if __name__ == "__main__":
    results = merge_results()
    #print_result_stats(results)
    #plot_heatmap(results)
    get_table_values(results)
