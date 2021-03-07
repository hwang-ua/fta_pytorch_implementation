import torch
import torch.nn as nn
import torch.nn.functional as functional


class FCBody(nn.Module):
    def __init__(self, input_dim, hidden_units=(64, 64), activation=functional.relu):
        super().__init__()
        dims = (input_dim,) + hidden_units
        self.layers = nn.ModuleList([layer_init_xavier(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.activation = activation
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            # print(layer(x).min(), layer(x).max())
            x = self.activation(layer(x))
        return x


class FCNetwork(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units, head_activation=None):
        super().__init__()
        body = FCBody(input_units, hidden_units=tuple(hidden_units))
        self.fc_head = layer_init_xavier(nn.Linear(body.feature_dim, output_units))
        self.to(device)

        self.device = device
        self.body = body
        self.head_activation = head_activation

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        phi = self.body(x)
        phi = self.fc_head(phi)
        if self.head_activation is not None:
            phi = self.head_activation(phi)
        return phi


def layer_init_xavier(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x
