import math

from torch import nn

class MLP(nn.Module):
    """Standard feed-forward MLP used for regression tasks.

    Builds a sequence of fully-connected layers with ReLU activations and
    dropout, terminating in a linear output layer.

    Args:
        input_dim (int): Dimensionality of input features.
        output_dim (int): Dimensionality of outputs.
        hidden_dim (int): Width of hidden layers.
        n_layers (int): Number of hidden layers.
        dropout_rate (float): Dropout probability applied after each hidden layer.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout_rate):
        super(MLP, self).__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers

        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-1], output_dim))

        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor of shape (batch, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch, output_dim).
        """
        return self.net(x)

def kaiming_init(model):
    """Initialize model parameters using a Kaiming-like scheme.

    This function iterates named parameters and applies heuristic initial
    values to biases and weights, helping stable training for MLPs.

    Args:
        model (nn.Module): Model whose parameters will be initialized in-place.
    """
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.startswith("layers.0"):
            param.data.normal_(0, 1/math.sqrt(param.shape[1]))
        else:
            param.data.normal_(0, math.sqrt(2)/math.sqrt(param.shape[1]))