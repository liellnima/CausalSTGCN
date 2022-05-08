import torch
import json
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# shape information:

# graph shape: timewindow, features, number_of_neighbors * features
# number of neighbours (only past) with radius 2 in 1d world: 5 (2 above + 2 below + itself)

# TODO handle other type of datasets!!! (with other shapes) (check if it works without adaption)

class GridcellDataset(Dataset):
    """ Costum torch dataset for our synthetic data.
    """
    def __init__(self, X, A, Y, Y_baseline):
        """
        Params:
            X (torch.tensor <N, T, D, G>): Input data with N samples, T timesteps,
                D featurse and G gridcells
            A (torch.tensor <T, D, D, G, G>): Adjacency matrix between features
                and gridcells over time
            Y (torch.tensor <N, T, P, G>): Output data with N samples, time-series
             of T, one prediction dimension P and G gridcells
             (-> time series regression on nodes)
        """
        self.X = X
        self.A = A
        self.Y = Y
        self.Y_baseline = Y_baseline

    def __len__(self):
        """ Returns number of samples
        """
        return self.X.shape[0]

    def __getitem__(self, idx):
        """ Return a sample of our dataset.
        Params:
            idx (int): index indicating which sample we want
        Returns:
            tuple <tuple <torch.tensor, torch.tensor>, torch.tensor> :
                Tuple with input data and output data. Input data contains both
                node and adj matrix
        """
        x_item = self.X[idx, :, :, :]
        a_item = self.A  # the same for all samples
        y_item = self.Y[idx, :, :, :]
        y_baseline = self.Y_baseline[idx, :, :, :]

        return (x_item, a_item, y_baseline), y_item


class DataProcesser:
    """ Preprocesses the data for the STGCN. Loads the data, creates torch
    data loaders and this class can be extended for additional preprocessing.
    """
    def __init__(self, path):
        """
        Loads the data, generates tensors from it and calculates the adj matrix.
        With this information we can create torch data loaders afterwards.
        Params:
            path (str): tells us where the synthetic data is stored
        """
        # shape of x: samples, timewindow, features, gridcells
        data_x, data_y, dcdi_x, dcdi_y, dcdi_weights, data_params = self.load_data(Path(path))
        self.dcdi_weights = dcdi_weights
        self.data_x = torch.from_numpy(data_x) # just for completeness
        self.num_gridcells = data_params["num_gridcells"]
        self.radius_neighbors = data_params["neighborhood"]

        # create ground truth data
        self.Y = torch.from_numpy(data_y)

        # create adj matrix (the same for all batches rn)
        self.adj_matrix = self.create_adj_matrix(self.dcdi_weights, self.num_gridcells, self.radius_neighbors)

        # create x data (batch_size, features, time, grid_cells)  (N, C, T, V)
        self.X = torch.from_numpy(np.concatenate((dcdi_x, dcdi_y), axis=2))

        # create tensor with predictions from dcdi (baseline)
        self.baseline_pred = torch.from_numpy(dcdi_y)

        # scale data and create the right dimension orderings of X and Y
        self.scale_data()
        self.permute_X_Y()

    def get_data(self):
        """ Returns all data items.
        Returns:
            tuple <torch.tensors>: X, Y, adj_matrix, dcdi_weights, baseline_pred
        """
        return self.X, self.Y, self.adj_matrix, self.dcdi_weights, self.baseline_pred

    # TODO maybe also normalize?
    def scale_data(self):
        """ Scale the given data
        Note: dcdi weights are already between -1 and 1
        """
        # the observed min max values are - 10 and 10
        # we can just divide everything by 10 to scale the data between -1 and 1
        self.X = torch.div(self.X, 10)
        self.Y = torch.div(self.Y, 10)
        self.baseline_pred = torch.div(self.baseline_pred, 10)
        self.data_x = torch.div(self.data_x, 10)

    # TODO: we need train, tune and valid dataset
    def get_dataloaders(self, batch_size=32, train_valid_split=0.8):
        """ Can be called to get a train and valid dataloader for pytorch
        Params:
            batch_size (int): which batch size we want to have
            train_valid_split (float): how much data should be for training
        Returns:
            Tuple (torch.DataLoader, torch.DataLoader): the data loaders
        """
        # create costum torch dataset
        dataset = GridcellDataset(self.X, self.adj_matrix, self.Y, self.baseline_pred)

        # split into train and validation data
        train_len = int(train_valid_split * len(dataset))
        valid_len = len(dataset) - train_len
        train_set, valid_set = random_split(dataset, (train_len, valid_len), generator=torch.Generator().manual_seed(42))

        # create data loaders
        self.loader_train = DataLoader(
            train_set,
            batch_size = batch_size,
            shuffle = False,
            num_workers = 2,
            pin_memory = (torch == "cuda"),
            )
        self.loader_valid = DataLoader(
            valid_set,
            batch_size = batch_size,
            shuffle = False,
            num_workers = 2,
            pin_memory = (torch == "cuda"),
            )

        return self.loader_train, self.loader_valid

    def create_adj_matrix(self, weights, num_gridcells, radius_neighbors):
        """ Creates an adjacency matrix from a given set of weights
        Params:
            weights (np.array): shape of <timewindow, features, num_neighbors x features>
            num_gidcells (int): how many gridcells are used
            radius_neighbors (int): neighbours radius taken into account

        Returns: torch.tensor with shape:
            <Time, Feature, Feature, Gridcell, Gridcell>
            This is the adj matrix for our graph with inter-time connections
        """
        # get the relevant numbers
        num_neighbors = radius_neighbors * 2 + 1 # only for 1d case times 2, 2d: x 4
        dim_t, dim_d, dim_neighborhood = weights.shape

        # create empty array with right shape
        adj_matrix = np.zeros((dim_t+1, dim_d, dim_d, num_gridcells, num_gridcells))

        # access the temporal edges of the gridcells - loop through time and features
        for d in range(dim_d):

            # first timestep remains zeros, since we have no instantenous connections rn
            for t in range(1, dim_t+1):

                # get neighbours of g (being d) from t steps before
                neighbors = weights[t-1, d, :]
                # split neighbours; structure: (g<d,d>, g<d,d>, g<d,d>)
                neighbors = np.split(neighbors, dim_d)

                for g in range(num_gridcells):

                    # assign neighbors
                    for i_neighbor, d_n_weights in enumerate(neighbors):

                        # calculate which grid cells we are looking at
                        neighbor_grid_pos = g - radius_neighbors + i_neighbor
                        # skip neighbors that are outside of the grid and cannot exist (edge cases)
                        if (neighbor_grid_pos >= 0) and (neighbor_grid_pos < num_gridcells):
                            adj_matrix[t, d, :, g, neighbor_grid_pos] = d_n_weights

        # for example: how does influence feature 0 at location 5, the feature 1 at location 6?
        # print(adj_matrix[:, 0, 1, 5, 6]) (through complete time!)

        # LATER: differentiate between time window and time length, find some solution for that

        return torch.from_numpy(adj_matrix).float()

    def permute_X_Y(self):
        """ Permutes X and Y to get the right dim orderings
        """
        self.X = torch.permute(self.X, (0, 2, 1, 3))
        self.Y = torch.permute(self.Y, (0, 2, 1, 3))
        self.baseline_pred = torch.permute(self.baseline_pred, (0, 2, 1, 3))

    def load_data(self, path: Path):
        """ Loads data that was stored during data generation
        :param path
        """
        x = np.load(path / "data_x.npy")
        y = np.load(path / "data_y.npy")
        dcdi_x = np.load(path / "dcdi_data_x.npy")
        dcdi_y = np.load(path / "dcdi_data_y.npy")
        dcdi_weights = np.load(path / "dcdi_weights.npy")

        # load param dict
        with open(path / "data_params.json") as file:
            data_params = json.load(file)

        return x, y, dcdi_x, dcdi_y, dcdi_weights, data_params
