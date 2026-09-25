import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """Simple feed-forward feature extractor.

    Constructs a sequential MLP with `n_layers` hidden layers of dimension
    `hidden_dim` using ReLU activations. Used to transform raw inputs into
    a representation consumed by downstream Bayesian layers.

    Args:
        input_dim (int): Dimensionality of input features.
        hidden_dim (int): Width of each hidden layer.
        n_layers (int): Number of hidden layers.
    """
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers

        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the feature extractor.

        Args:
            x (Tensor): Input tensor of shape (batch, input_dim).

        Returns:
            Tensor: Extracted features of shape (batch, hidden_dim).
        """
        return self.net(x)    

class BayesianLinear(PyroModule):
    """Bayesian linear layer using Pyro priors.

    This layer places Normal priors over the weight matrix and bias vector and
    exposes the same call semantics as a regular linear layer.

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        prior_scale (float): Standard deviation of the Normal prior.
    """
    def __init__(self, in_features, out_features, prior_scale=1.0):
        super().__init__()

        self.weight = PyroSample(
            dist.Normal(0., prior_scale)
            .expand([out_features, in_features])
            .to_event(2)
        )

        self.bias = PyroSample(
            dist.Normal(0., prior_scale)
            .expand([out_features])
            .to_event(1)
        )

    def forward(self, x):
        """Compute linear transformation `x W^T + b`.

        Args:
            x (Tensor): Input tensor of shape (batch, in_features).

        Returns:
            Tensor: Output tensor of shape (batch, out_features).
        """
        return x @ self.weight.T + self.bias