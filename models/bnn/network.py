import math
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from models.bnn.bnn_components import FeatureExtractor, BayesianLinear

class PartialBNN(PyroModule):

    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, prior_scale, dataset_size):
        super().__init__()

        self.dataset_size = dataset_size

        self.features = FeatureExtractor(input_dim, hidden_dim, n_layers)
        self.out = BayesianLinear(hidden_dim, output_dim, prior_scale)

    def forward(self, x, y=None):
        h = self.features(x)
        mu = self.out(h)

        dim = mu.shape[-1]
        rank = min(3, dim)
        cov_factor = pyro.sample("cov_factor", dist.Normal(0., 1.0).expand([dim, rank]).to_event(2))
        cov_diag = pyro.sample("cov_diag", dist.HalfNormal(torch.ones(dim)).to_event(1))

        with pyro.plate("data", size = self.dataset_size,  subsample_size = x.shape[0]):
            obs = pyro.sample("obs", dist.LowRankMultivariateNormal(loc=mu, cov_factor=cov_factor, cov_diag=cov_diag), obs=y)
        return mu

class FullBNN(PyroModule):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, prior_scale, dataset_size):
        super().__init__()

        self.dataset_size = dataset_size
        self.activation = nn.Tanh()

        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        layer_list = [PyroModule[nn.Linear](dims[i-1], dims[i]) for i in 
                      range(1, len(dims))]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(0, prior_scale * math.sqrt(1 / dims[layer_idx])).expand(
                [dims[layer_idx +1], dims[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0,prior_scale * math.sqrt(1 / dims[layer_idx])).expand(
                [dims[layer_idx +1]]).to_event(1))
            
    def forward(self, x, y=None):
        x = self.activation(self.layers[0](x))
        for layer in self.layers[1:-1]:
             x = self.activation(layer(x))
        mu = self.layers[-1](x)

        dim = mu.shape[-1]
        rank = min(3, dim)
        cov_factor = pyro.sample("cov_factor", dist.Normal(0., 1.0).expand([dim, rank]).to_event(2))
        cov_diag = pyro.sample("cov_diag", dist.HalfNormal(torch.ones(dim)).to_event(1))

        with pyro.plate("data", size = self.dataset_size,  subsample_size = x.shape[0]):
            obs = pyro.sample("obs", dist.LowRankMultivariateNormal(loc=mu, cov_factor=cov_factor, cov_diag=cov_diag), obs=y)
        return mu