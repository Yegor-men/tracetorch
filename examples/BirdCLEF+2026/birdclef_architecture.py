import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

min_timescale = tt.functional.halflife_to_decay(1)
max_timescale = tt.functional.halflife_to_decay(62.5)
timescale_diff = max_timescale - min_timescale


class ResidualTNG(tt.Model):
    def __init__(self, working_dim, num_neurons, num_connections):
        super().__init__()

        self.lin_in = nn.Linear(working_dim, working_dim, bias=False)
        nn.init.normal_(self.lin_in.weight, 0, 0.1)

        self.tng = tt.nn.TopologicalNeuralGraph(
            neurons=tt.snn.DLIB(
                num_neurons,
                pos_beta=torch.rand(num_neurons),
                neg_beta=torch.rand(num_neurons),
                threshold=torch.rand(num_neurons),
            ),
            in_features=working_dim,
            out_features=working_dim,
            avg_out_degree=num_connections,
        )

        self.lin_out = nn.Linear(working_dim, working_dim, bias=False)
        nn.init.normal_(self.lin_out.weight, 0, 0.1)

    def forward(self, x):
        residual = x
        x = self.lin_in(x)
        x = self.tng(x)
        x = self.lin_out(x)
        return residual + x


class BirdClefModel(tt.Model):
    def __init__(
            self,
            in_features=768,
            working_dim=1024,
            num_neurons=1024,
            num_connections=16,
            num_layers=5,
            num_classes=234,
    ):
        super().__init__()
        self.enc = nn.Linear(in_features, working_dim, bias=False)

        self.layers = nn.ModuleList([ResidualTNG(
            working_dim=working_dim,
            num_neurons=num_neurons,
            num_connections=num_connections,
        ) for _ in range(num_layers)])

        self.ssm = tt.ssm.Mamba(working_dim, 16)

        self.dec = nn.Linear(working_dim, num_classes, bias=False)
        nn.init.normal_(self.dec.weight, 0, 0.03)

    def forward(self, x):
        x = self.enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ssm(x)
        x = self.dec(x)
        return x
