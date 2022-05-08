###### Author of this code is mainly Philippe Brouillard (@kurowasan)#####
import os
import torch
import json
import torch.nn as nn
import torch.distributions as distr
import numpy as np
from collections import OrderedDict

from tqdm import tqdm


class DataGeneratorWithLatent:
    """
    Code use to generate synthetic data with latent variables
    """

    # TODO: add instantenous relations
    def __init__(self, hp):
        self.hp = hp
        self.n = hp.n
        self.d = hp.num_features
        self.d_x = hp.num_gridcells
        self.tau = hp.timewindow

        if self.n > 1:
            self.t = self.tau + 1
        else:
            self.t = hp.num_timesteps

        self.k = hp.num_clusters
        self.prob = hp.prob
        self.noise_coeff = hp.noise_coeff

        self.num_layers = hp.num_layers
        self.num_hidden = hp.num_hidden
        self.non_linearity = hp.non_linearity
        self.same_cluster_assign = True

        assert self.d_x > self.k, f"dx={self.d_x} should be larger than k={self.k}"

    def save_data(self, path):
        with open(os.path.join(path, "data_params.json"), "w") as file:
            json.dump(vars(self.hp), file, indent=4)
        np.save(os.path.join(path, "data_x"), self.X.detach().numpy())
        np.save(os.path.join(path, "data_z"), self.Z.detach().numpy())
        np.save(os.path.join(path, "graph"), self.G.detach().numpy())
        np.save(os.path.join(path, "graph_w"), self.w.detach().numpy())

    def sample_graph(self) -> torch.Tensor:
        """
        Sample a random matrix that will be used as an adjacency matrix
        The diagonal is set to 1.
        Returns:
            A Tensor of tau graphs between the Z (shape: tau x (d x k) x (d x k))
        """
        prob_tensor = (
            torch.ones((self.tau, self.d * self.k, self.d * self.k)) * self.prob
        )
        prob_tensor[
            :, torch.arange(prob_tensor.size(1)), torch.arange(prob_tensor.size(2))
        ] = 1

        G = torch.bernoulli(prob_tensor)

        return G

    def sample_mlp(self):
        """Sample a MLP that outputs the parameters for the distributions of Z"""
        dict_layers = OrderedDict()
        num_first_layer = self.tau * (self.d * self.k) * (self.d * self.k)
        num_last_layer = 2 * self.d * self.k

        if self.non_linearity == "relu":
            nonlin = nn.ReLU()
        else:
            raise NotImplementedError("Nonlinearity is not implemented yet")

        for i in range(self.num_layers + 1):
            num_input = self.num_hidden
            num_output = self.num_hidden

            if i == 0:
                num_input = num_first_layer
            if i == self.num_layers:
                num_output = num_last_layer
            dict_layers[f"lin{i}"] = nn.Linear(num_input, num_output)

            if i != self.num_layers:
                dict_layers[f"nonlin{i}"] = nonlin

        f = nn.Sequential(dict_layers)

        return f

    def sample_lstm(self):
        pass

    def sample_w(self) -> torch.Tensor:
        """Sample matrices that are positive and orthogonal.
        They are the links between Z and X.
        Returns:
            A tensor w (shape: d_x, d, k)
        """
        # assign d_xs uniformly to a cluster k
        cluster_assign = np.random.choice(self.k, size=self.d_x - self.k)
        cluster_assign = np.append(cluster_assign, np.arange(self.k))
        cluster_assign = np.stack((np.arange(self.d_x), cluster_assign))

        # sample w uniformly and mask it according to the cluster assignment
        mask = torch.zeros((self.d_x, self.k))
        mask[cluster_assign] = 1
        w = torch.empty((self.d_x, self.k)).uniform_(0.5, 2)
        w = w * mask

        # shuffle rows
        w = w[torch.randperm(w.size(0))]

        # normalize to make w orthonormal
        w = w / torch.norm(w, dim=0)

        if self.same_cluster_assign:
            w = w.unsqueeze(1).repeat(1, self.d, 1)
        else:
            raise NotImplementedError("This type of w sampling is not implemented yet")

        # TODO: add test torch.matmul(w.T, w) == torch.eye(w.size(1))
        return w

    def generate(self):
        """Main method to generate data
        Returns:
            X, Z, respectively the observable data and the latent
        """
        # initialize Z for the first timesteps
        self.Z = torch.zeros((self.t, self.d, self.k))
        self.X = torch.zeros((self.t, self.d, self.d_x))
        for i in range(self.tau):
            self.Z[i].normal_(0, 1)

        # sample graphs and NNs
        self.G = self.sample_graph()
        self.f = self.sample_mlp()

        # sample the latent Z
        for t in range(self.tau, self.t):
            g = self.G.view(self.G.shape[0], -1)
            z = self.Z[t - self.tau : t].view(self.tau, -1).repeat(1, self.d * self.k)
            nn_input = (g * z).view(-1)
            params = self.f(nn_input).view(-1, 2)
            params[:, 1] = 1 / 2 * torch.exp(params[:, 1])
            dist = distr.normal.Normal(params[:, 0], params[:, 1])
            self.Z[t] = dist.rsample().view(self.d, self.k)

        # sample observational model
        self.w = self.sample_w()

        # sample the data X
        for t in range(self.tau, self.t):
            mean = torch.einsum("xdk,dk->dx", self.w, self.Z[t])
            # could sample sigma
            dist = distr.normal.Normal(mean.view(-1), 1)
            self.X[t] = dist.rsample().view(self.d, self.d_x)

        return self.X, self.Z


class DataGeneratorWithoutLatent:
    """
    Code use to generate synthetic data without latent variables
    """

    # TODO: add instantenous relations
    def __init__(self, hp):
        self.hp = hp
        self.n = hp.n
        self.d = hp.num_features
        self.d_x = hp.num_gridcells
        self.world_dim = hp.world_dim
        self.tau = hp.timewindow
        self.tau_neigh = hp.neighborhood
        self.prob = hp.prob
        self.eta = hp.eta
        self.noise_coeff = hp.noise_coeff

        if self.n > 1:
            self.t = self.tau + 1
        else:
            self.t = hp.num_timesteps

        self.num_layers = hp.num_layers
        self.num_hidden = hp.num_hidden
        self.non_linearity = hp.non_linearity

        assert hp.world_dim <= 2 and hp.world_dim >= 1, "world_dim not supported"
        self.num_neigh = (self.tau_neigh * 2 + 1) ** self.world_dim

    def save_data(self, path):
        with open(os.path.join(path, "data_params.json"), "w") as file:
            json.dump(vars(self.hp), file, indent=4)
        np.save(os.path.join(path, "data_x"), self.X.detach().numpy())
        np.save(os.path.join(path, "graph"), self.G.detach().numpy())
        np.save(os.path.join(path, "weights"), self.weights.detach().numpy())

        # optional saves
        if hasattr(self, "dcdi_X"):
            np.save(os.path.join(path, "dcdi_data_x"), self.dcdi_X.detach().numpy())
        if hasattr(self, "Y"):
            np.save(os.path.join(path, "data_y"), self.Y.detach().numpy())
        if hasattr(self, "dcdi_Y"):
            np.save(os.path.join(path, "dcdi_data_y"), self.dcdi_Y.detach().numpy())
        if hasattr(self, "dcdi_weights"):
            np.save(os.path.join(path, "dcdi_weights"), self.dcdi_weights.detach().numpy())


    def sample_graph(self, diagonal=False) -> torch.Tensor:
        """
        Sample a random matrix that will be used as an adjacency matrix
        The diagonal is set to 1.
        Returns:
            A Tensor of tau graphs, size: (tau, d, num_neighbor x d)
        """
        # TODO: allow data with any number of dimension (1D, 2D, ...)
        prob_tensor = (
            torch.ones((self.tau, self.d, self.num_neigh * self.d)) * self.prob
        )

        if diagonal:
            # set diagonal to 1
            prob_tensor[
                :, torch.arange(prob_tensor.size(1)), torch.arange(prob_tensor.size(2))
            ] = 1

        G = torch.bernoulli(prob_tensor)

        return G

    def sample_linear_weights( # lower: 0.1, upper: 0.3
        self, lower: int = 0.3, upper: float = 0.5, eta: float = 1
    ) -> torch.Tensor:
        """Sample the coefficient of the linear relations
        :param lower: lower bound of uniform distr to sample from
        :param upper: upper bound of uniform distr
        :param eta: weight decay parameter, reduce the influences of variables
        that are farther back in time, should be >= 1
        """
        # TODO: remove, only for tests
        # if self.G.shape[1] == 2:
        #     weights = torch.empty_like(self.G).uniform_(lower, upper)
        #     # known periodic linear dynamical system
        #     weights[0] = torch.tensor([[-3.06, 1.68],
        #                                [-4.20, 1.97]])
        #     # weights[0] = torch.tensor([[-1.72, -3.91],
        #     #                            [1.56, 2.97]])
        #     self.G[0] = torch.tensor([[1, 1],
        #                              [1, 1]])
        # else:
        sign = torch.ones_like(self.G) * 0.5
        sign = torch.bernoulli(sign) * 2 - 1
        weights = torch.empty_like(self.G).uniform_(lower, upper)
        weights = sign * weights * self.G
        weight_decay = 1 / torch.pow(eta, torch.arange(self.tau))
        weights = weights * weight_decay.view(-1, 1, 1)

        return weights

    def split_target_data(self, target_idx: int = -1):
        """Separates target / y-data from input / x-data.
        :param target_idx: which feature index should be used as target
        """
        # split of target data from X
        splits = torch.tensor_split(self.X, self.d, dim=2)
        self.Y = splits[target_idx]
        if (target_idx == -1) or (target_idx == len(splits)):
            x_data = splits[:target_idx]
        elif target_idx == 0:
            x_data = splits[target_idx + 1 :]
        else:
            x_data = splits[:target_idx] + splits[target_idx + 1 :]

        self.X = torch.concat(x_data, dim=2)

        # same for dcdi data
        if hasattr(self, "dcdi_X"):
            splits = torch.tensor_split(self.dcdi_X, self.d, dim=2)
            self.dcdi_Y = splits[target_idx]
            if (target_idx == -1) or (target_idx == len(splits)):
                x_data = splits[:target_idx]
            elif target_idx == 0:
                x_data = splits[target_idx + 1 :]
            else:
                x_data = splits[:target_idx] + splits[target_idx + 1 :]

            self.dcdi_X = torch.concat(x_data, dim=2)

    def pertubate_weights(
        self, weights: torch.Tensor, pertubation_rate: float = 0.2
    ) -> torch.Tensor:
        """Pertubates a given weight tensor.
        :param weight: Weight tensor that should be pertubated
        :param pertubation_rate: how many percent of the weights should be
        dropped or changed
        """
        # number of entries that should be pertubated
        entries = torch.numel(weights) * (pertubation_rate / 2)

        # values close to zero are set to zero
        close_to_zero_mask = (weights < 0.1) & (weights > -0.1) & (weights != 0)
        # set enough / not too many values to zero
        zero_entries = torch.sum(close_to_zero_mask)
        if zero_entries > entries:
            num_set_false = int(zero_entries - entries)
            indices = torch.randint(torch.numel(close_to_zero_mask), (num_set_false,))
            close_to_zero_mask.flatten()[indices] = 1
            close_to_zero_mask = torch.reshape(close_to_zero_mask, weights.size())
        else:
            num_set_true = int(entries - zero_entries)
            indices = torch.randint(torch.numel(close_to_zero_mask), (num_set_true,))
            close_to_zero_mask.flatten()[indices] = 1
            close_to_zero_mask = torch.reshape(close_to_zero_mask, weights.size())

        weights[close_to_zero_mask] = 0.0

        # the rest gets random numbers
        pertubated_weights = weights
        indices = torch.randint(torch.numel(weights), (int(entries),))
        pertubated_weights.flatten()[indices] = torch.FloatTensor(
            np.random.uniform(size=int(entries))
        )
        weights = torch.reshape(pertubated_weights, weights.size())

        return weights

    def generate_dcdi_data(self, pertubation_rate: float = 0.2) -> torch.Tensor:
        """Optional method to generate a simulated DCDI graph and
        its predicted data. Can only be run after generate function was run.
        :param pertubation_rate: how many percent of the weights should be
        dropped or changed
        """
        self.dcdi_X = torch.zeros((self.n, self.t, self.d, self.d_x))

        # adapt the weight matrix - dropout and change weights
        self.dcdi_weights = self.weights.detach().clone()
        self.dcdi_weights = self.pertubate_weights(self.dcdi_weights, pertubation_rate)

        print("Generating the DCDI data ...")
        for i_n in tqdm(range(self.n)):
            # initialize X and get the exact same original noise again
            self.dcdi_X[i_n, :self.tau] = self.noise[i_n, :self.tau]

            for t in range(self.tau, self.t):
                for i in range(self.d_x):
                    lower_x = max(0, i - self.tau_neigh)
                    upper_x = min(self.dcdi_X.size(-1) - 1, i + self.tau_neigh) + 1
                    lower_w = max(0, i - self.tau_neigh) - (i - self.tau_neigh)
                    upper_w = (
                        min(self.dcdi_X.size(-1) - 1, i + self.tau_neigh) - i + self.tau_neigh + 1
                    )

                    if self.d_x == 1:
                        w = self.dcdi_weights[:, :, : self.d]
                        x = self.dcdi_X[i_n, t - self.tau:t, :, :self.d].reshape(self.tau, -1)
                    else:
                        w = self.dcdi_weights[:, :, lower_w * self.d: upper_w * self.d]
                        x = self.dcdi_X[i_n, t - self.tau:t, :, lower_x:upper_x].reshape(self.tau, -1)

                    self.dcdi_X[i_n, t, :, i] = (
                        torch.einsum("tij,tj->i", w, x)
                        + self.noise_coeff * self.noise[i_n, t, :, i]
                    )
        print("...finished.")

        # add small additional noise to everything (dcdi noise!)
        self.dcdi_X = torch.add(
            self.dcdi_X, (0.001**0.5) * torch.randn(self.dcdi_X.size())
        )

        return self.dcdi_X

    def generate(self) -> torch.Tensor:
        """Main method to generate data
        Returns:
            X, the data, size: (n, t, d, d_x)
        """
        # sample graphs and weights
        self.G = self.sample_graph()
        self.weights = self.sample_linear_weights(eta=self.eta)
        self.X = torch.zeros((self.n, self.t, self.d, self.d_x))
        self.noise = torch.normal(0, 1, size=self.X.size())

        print("Generating data ...")
        for i_n in tqdm(range(self.n)):
            # initialize X and sample noise
            self.X[i_n, :self.tau] = self.noise[i_n, :self.tau]

            for t in range(self.tau, self.t):
                # if self.d_x == 1:
                #     # TODO: only test
                #     x = self.X[t-1, :, 0]
                #     w = self.weights
                #     self.X[t, :, 0] = w[0].T @ x  # torch.einsum("tij,tij->i", w, x)
                # else:
                for i in range(self.d_x):
                    # TODO: should add wrap around
                    lower_x = max(0, i - self.tau_neigh)
                    upper_x = min(self.X.size(-1) - 1, i + self.tau_neigh) + 1
                    lower_w = max(0, i - self.tau_neigh) - (i - self.tau_neigh)
                    upper_w = min(self.X.size(-1) - 1, i + self.tau_neigh) - i + self.tau_neigh + 1

                    if self.d_x == 1:
                        w = self.weights[:, :, :self.d]
                        x = self.X[i_n, t - self.tau:t, :, :self.d].reshape(self.tau, -1)
                    else:
                        w = self.weights[:, :, lower_w * self.d: upper_w * self.d]
                        x = self.X[i_n, t - self.tau:t, :, lower_x:upper_x].reshape(self.tau, -1)

                    # w.size: (tau, d, d * (tau_neigh * 2 + 1))
                    # x.size: (tau, d * (tau_neigh * 2 + 1))
                    # print(w.size())
                    # print(x.size())
                    # __import__('ipdb').set_trace()
                    self.X[i_n, t, :, i] = torch.einsum("tij,tj->i", w, x) + self.noise_coeff * self.noise[i_n, t, :, i]
        print("... finished.")
        return self.X
