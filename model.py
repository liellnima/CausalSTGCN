# Code adapted from https://github.com/open-mmlab/mmskeleton
# based on the following paper: https://arxiv.org/pdf/1801.07455.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

# TODOs
# - adapt output (temporal node prediction, not a graph classification)
# - adapt temporal layer (more interconnections for our case)

class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,tddvv->nctv', (x, A))
        #x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class STGCNBlock(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.out_channels = out_channels

        # TODO might need to adapt this part here
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels).float(),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1),
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )

        #self.relu = nn.ReLU()
        #self.linear = nn.Linear()
        self.tanh = nn.Tanh()

    def forward(self, x, A):
        res = self.residual(x)

        # spatial convolutions
        x, A = self.gcn(x, A)

        # temporal convolutions
        x = self.tcn(x) + res
        x = self.tanh(x) # we want to get values between -1 and 1

        return x, A


class CausalSTGCN(nn.Module):
    def __init__(self, n_stgcn=1, n_txpcnn=1, input_feat=5, output_feat=1,
                 seq_len=101, pred_seq_len=101, kernel_size=3):
        """
        Params:
            n_stgcn (int): How many STGCN should be used (Graph Convolutions)
            n_txpcnn (int): How many temporal convolutions should be used
            input_feat (int): How many input features are provided
            output_feat (int): How many features should be predicted
            seq_len (int): Length of time-series input. This  is also the temporal
                kernel size. (Could be separated later, if time_window != len(time))
            pred_seq_len (int): how many timesteps should be predicted
            kernel_size (int): the spatial kernel size, use e.g. the area of neighbours,
                (num_neighbours*2 + 1)
        """
        super(CausalSTGCN, self).__init__()
        self.n_stgcn = n_stgcn
        self.n_txpcnn = n_txpcnn
        self.input_feat = input_feat
        self.output_feat = output_feat
        temporal_kernel_size = seq_len # our time-window
        spatial_kernel_size = kernel_size # should be num neighbours*2 + 1

        # spatial and temporal convolutions - kernel order was swapped for our case,
        # since our graph connections are on the temporal dimension
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(STGCNBlock(input_feat, output_feat, (spatial_kernel_size, temporal_kernel_size)))
        for j in range(1,self.n_stgcn):
            self.st_gcns.append(STGCNBlock(output_feat, output_feat, (spatial_kernel_size, temporal_kernel_size)))

        # temporal convolutions, time extrapolator
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len, 3, padding=1))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len, 3, padding=1))
        self.tpcnn_input = nn.Conv2d(input_feat, output_feat, 3, padding=1)
        self.tpcnn_output = nn.Conv2d(pred_seq_len,pred_seq_len, 3, padding=1)


        # activation functions
        self.tanhs = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.tanhs.append(nn.Tanh()) # nn.PReLU()


    def forward(self, x, A):
        """
        Params:
            v (torch.tensor): nodes of the graph with the shape:
                (N, C, T, V)
            a (torch.tensor): adjacency matrix of the graph with the shape:
                (T, C, C, V, V) (features C and nodes/gridcelss V)
        """
        # STGCN BLOCKS

        for k in range(self.n_stgcn):
            print(x.dtype)
            print(A.dtype)
            exit(0)
            x, A = self.st_gcns[k](x, A)

        # TPCNN BLOCKS

        # replace first STGCN block and use only TPCNN
        if self.n_stgcn == 0:
            x = self.tpcnn_input(x)

        # replace TPCNN block and use only STGCN
        if self.n_txpcnn == 0:
            x = torch.permute(x, (0, 2, 1, 3))
            x = self.tpcnn_output(x)
            x = torch.permute(x, (0, 2, 1, 3))

        # TPCNN Blocks usual case
        else:
            x = torch.permute(x, (0, 2, 1, 3))
            x = self.tanhs[0](self.tpcnns[0](x))

            for k in range(1,self.n_txpcnn-1):
                x = self.tanhs[k](self.tpcnns[k](x)) + x

            x = self.tpcnn_output(x)
            x = torch.permute(x, (0, 2, 1, 3))

        return x, A
