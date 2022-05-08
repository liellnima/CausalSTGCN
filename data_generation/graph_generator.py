import argparse
import os
import torch

import numpy as np
from data_generation import DataGeneratorWithLatent, DataGeneratorWithoutLatent
from plot import plot_adjacency_graphs, plot_adjacency_w, plot_x, plot_z
# from causal.data_generation.data_generation import (
#     DataGeneratorWithLatent,
#     DataGeneratorWithoutLatent,
# )
# from causal.data_generation.plot import (
#     plot_adjacency_graphs,
#     plot_adjacency_w,
#     plot_x,
#     plot_z,
# )


def main(hp):
    """
    Args:
        hp: object containing hyperparameter values

    Returns:
        The observable data that has been generated
    """
    # Control as much randomness as possible
    torch.manual_seed(hp.random_seed)
    np.random.seed(hp.random_seed)

    if hp.latent:
        generator = DataGeneratorWithLatent(args)
    else:
        generator = DataGeneratorWithoutLatent(args)

    # Generate, save and plot data
    data = generator.generate()
    dcdi_data = generator.generate_dcdi_data()
    generator.split_target_data()

    # TODO: find out if this data is sufficent for our GNN approach??
    print("Saving ...")
    generator.save_data(hp.exp_path)
    print("Plotting ...")
    #plot_adjacency_graphs(generator.G, hp.exp_path)

    plot_x(generator.X.detach().numpy(), hp.exp_path)

    if hp.latent:
        plot_z(generator.Z.detach().numpy(), hp.exp_path)
        plot_adjacency_w(generator.w, hp.exp_path)

    return data


if __name__ == "__main__":
    # params I want to have
    # grid cells: 25
    # time window: 3, make 100
    # d: 5
    # no problem to set n > 1
    # what to run next: neighbourhood=2, g=20, timewindow=100, d=5, n=1000
    parser = argparse.ArgumentParser(
        description=" Usage to generate synthetic causal graphs to \
                                         test the causal-GNN idea."
    )

    parser.add_argument(
        "--exp-path", type=str, default="../dataset/", help="Path to experiments"
    )
    parser.add_argument(
        "--exp-id", type=int, default=0, help="ID unique to the dataset"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=3,
        help="Random seed used for torch and numpy",
    )

    # Dataset properties
    parser.add_argument(
        "--latent", action="store_true", help="Use generative model with latents"
    )
    parser.add_argument(# n = 10000
        "--n", type=int, default=1, help="Number of time-series"
    )  # J: What to do you mean by that?
    parser.add_argument(
        "-t",
        "--num-timesteps",
        type=int,
        default=10000, # is overridden if n>1
        help="Number of timesteps in total",
    )
    parser.add_argument(
        "-d", "--num-features", type=int, default=2, help="Number of features"
    )
    parser.add_argument(
        "-g",
        "--num-gridcells",
        type=int,
        default=10,  # J: more grid cells solve the problem!
        help="Number of gridcells",  # J: Total number of grid cells
    )
    parser.add_argument(
        "-k", "--num-clusters", type=int, default=3, help="Number of clusters"
    )
    parser.add_argument(
        "-p",
        "--prob",
        type=float,
        default=0.6,
        help="Probability of an edge in the causal graphs",
    )  # J: change for sparse and dense graph and compare
    parser.add_argument(
        "--noise-coeff",
        type=float,
        default=1,
        help="Coefficient for the additive noise",
    )
    parser.add_argument(
        "--world-dim",
        type=float,
        default=1,
        help="Number of dimension for the gridcell (1D or 2D)",
    )  # J: Only 1d for the moment! Can we also do 3d?
    parser.add_argument(
        "--circular-padding",
        action="store_true",
        help="If true, pad gridcells in a circular fashion",
    )  # J: not possible yet
    parser.add_argument(
        "--neighborhood",
        type=int,
        default=1,
        help="'Radius' of neighboring gridcells that have an influence",
    )
    parser.add_argument(
        "--timewindow",
        type=int,
        default=1,  # J: put this to the maximum time scale if you put n > 1
        help="Number of previous timestep that interacts with a timestep t",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.5,
        help="Weight decay applied to linear coefficients",
    )  # J: the higher eta, the less we look at the past

    # Neural network (NN) architecture
    parser.add_argument(
        "--num-layers", type=int, default=1, help="Number of layers in NN"
    )
    parser.add_argument(
        "--num-hidden", type=int, default=4, help="Number of hidden units in NN"
    )
    parser.add_argument(
        "--non-linearity",
        type=str,
        default="relu",
        help="Type of non-linearity used in the NN",
    )

    args = parser.parse_args()

    # Create folder
    args.exp_path = os.path.join(args.exp_path, f"data{args.exp_id}")
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    main(args)
