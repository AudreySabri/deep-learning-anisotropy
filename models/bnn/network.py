import math
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from models.bnn.bnn_components import FeatureExtractor, BayesianLinear

class PartialBNN(PyroModule):
    """Partial Bayesian neural network that places priors on final layer.

    This model uses a deterministic feature extractor followed by a
    `BayesianLinear` output layer. Observations are modeled with a
    LowRankMultivariateNormal likelihood whose parameters are sampled inside
    the forward pass so that SVI can learn posterior distributions.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Dimension of model outputs.
        hidden_dim (int): Hidden layer dimension for the feature extractor.
        n_layers (int): Number of hidden layers in the feature extractor.
        prior_scale (float): Scale (std) of Normal priors on weights.
        dataset_size (int): Total dataset size (used for plates/subsampling).
    """
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, prior_scale, dataset_size):
        super().__init__()

        self.dataset_size = dataset_size

        self.features = FeatureExtractor(input_dim, hidden_dim, n_layers)
        self.out = BayesianLinear(hidden_dim, output_dim, prior_scale)

    def forward(self, x, y=None):
        """Forward pass that samples likelihood parameters and returns predictive mean.

        Args:
            x (Tensor): Input tensor of shape (batch, input_dim).
            y (Tensor, optional): Optional target values for conditioning the
                observed site during training.

        Returns:
            Tensor: Predictive mean `mu` of shape (batch, output_dim).
        """
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
    """Fully Bayesian neural network placing priors on all linear layers.

    Builds an MLP where each linear layer's weights and biases are Pyro samples
    with Normal priors. Observations are modeled with a
    LowRankMultivariateNormal likelihood as in `PartialBNN`.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Output dimensionality.
        hidden_dim (int): Hidden layer width.
        n_layers (int): Number of hidden layers.
        prior_scale (float): Scale (std) for Normal priors.
        dataset_size (int): Dataset size for the Pyro plate.
    """
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
        """Forward pass through the fully Bayesian MLP.

        Args:
            x (Tensor): Input tensor of shape (batch, input_dim).
            y (Tensor, optional): Optional targets to condition the observed site.

        Returns:
            Tensor: Predictive mean of shape (batch, output_dim).
        """
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