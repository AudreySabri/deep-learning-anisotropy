import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers

        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)    

class BayesianLinear(PyroModule):
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
        return x @ self.weight.T + self.bias