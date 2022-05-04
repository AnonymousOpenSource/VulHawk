import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import LayerNorm, Linear, ReLU
from torch_scatter import scatter
import os
import torch
from utils.gcn2_conv import GCN2Conv
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch.nn import init
import math

class MyData(Data):
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos)

class AdapterHead(torch.nn.Module):
    def __init__(self, meta_dim, hidden_dim):
        super().__init__()
        self.device = torch.device("cuda:0")
        self.layer_norm_meta = LayerNorm(meta_dim)
        self.layer_norm = LayerNorm(hidden_dim)
        # self.down_project = Parameter(torch.Tensor(meta_dim, hidden_dim))
        self.down_project = Linear(meta_dim, hidden_dim, bias=False)
        self.up_project = Linear(hidden_dim, hidden_dim, bias=False)
        self.layer_norm_input = LayerNorm(hidden_dim)


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input_x=None, input_entropy=None):
        x = input_x.to(self.device)
        entropy = input_entropy.to(self.device)
        normalized_entropy = self.layer_norm_meta(entropy)
        normalized_x = x
        h = F.relu(F.relu(self.down_project(normalized_entropy)) * normalized_x)
        return self.up_project(h) + x

class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()
        self.device = torch.device("cuda:0")
        self.lins = torch.nn.ModuleList()
        self.graph_pooling_type = "average"
        self.lins.append(Linear(64, hidden_channels))
        self.lins.append(Linear(hidden_channels, 64))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
        self.adapter = AdapterHead(3, hidden_channels)
        self.dropout = dropout

    def __preprocess_graphpool(self, batch_graph):
        start_idx = batch_graph.ptr
        idx = []
        elem = []
        for i, sidx in enumerate(batch_graph.ptr[1:]):

            if self.graph_pooling_type == "average":
                # average pooling
                elem.extend([1. / (sidx - start_idx[i])] * (sidx - start_idx[i]))
            else:
                # sum pooling
                elem.extend([1] * (sidx - start_idx[i]))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph.ptr) - 1, start_idx[-1]]))

        return graph_pool.to(self.device)

    def forward(self, input_x=None, edge_index=None):
        if isinstance(input_x, Tensor) and isinstance(edge_index, Tensor):
            x = input_x
            x = F.dropout(x, self.dropout, training=self.training)
            x = x_0 = self.lins[0](x).relu()

            for conv in self.convs:
                x = F.dropout(x, self.dropout, training=self.training)
                x = conv(x, x_0, edge_index)
                x = x.relu()
                # x = F.gelu(x)

            x = F.dropout(x, self.dropout, training=self.training)
            x = self.lins[1](x)
            if self.graph_pooling_type == "average":
                return x.mean(dim=0)
            else:
                return x.sum(dim=0)
            # return x.log_softmax(dim=-1)
        else:
            x = input_x.x.to(self.device)
            edge_index = input_x.edge_index.to(self.device)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x_0 = self.lins[0](x).relu()

            for conv in self.convs:
                x = F.dropout(x, self.dropout, training=self.training)
                x = conv(x, x_0, edge_index)
                x = x.relu()

            x = F.dropout(x, self.dropout, training=self.training)
            x = self.lins[1](x)
            graph_pool = self.__preprocess_graphpool(input_x)
            pooled_x = torch.spmm(graph_pool, x)
            if getattr(self, "adapter", None):
                entropy = input_x.edge_attr.to(self.device).reshape(len(input_x.edge_attr) // 3, 3)
                pooled_x = self.adapter(pooled_x, entropy)
            if self.training:
                return pooled_x
            else:
                return pooled_x.detach()
