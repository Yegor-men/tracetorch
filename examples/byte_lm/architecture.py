import torch
from torch import nn
import tracetorch as tt


class Bar(tt.Model):
    def __init__(self, working_dim, num_neurons):
        super().__init__()

        self.lin_in = nn.Linear(working_dim, working_dim, bias=False)
        nn.init.eye_(self.lin_in.weight)

        coordinates = torch.randn(num_neurons, 5)
        distances = torch.linalg.vector_norm(coordinates, ord=2, dim=-1)
        coordinates = coordinates / distances.unsqueeze(-1)

        out_degrees = torch.empty(num_neurons).log_normal_(2, 1) + 1

        flow_values = torch.ones_like(distances) + (5 / out_degrees)

        self.fdsr = tt.snn.FDSR(
            lif_neurons=tt.snn.LIB(
                num_neurons,
                beta=torch.rand(num_neurons),
                threshold=torch.rand(num_neurons),
                bias=torch.randn(num_neurons) * 0.1,
                quant_fn=nn.Identity(),
            ),
            coordinates=coordinates,
            flow_values=flow_values,
            out_degrees=out_degrees,
            in_features=working_dim,
            out_features=working_dim,
            dim=-1,
        )

        self.lin_out = nn.Linear(working_dim, working_dim, bias=False)
        nn.init.zeros_(self.lin_out.weight)

    def forward(self, x):
        return x + self.lin_out(self.fdsr(self.lin_in(x)))


class TTLM(tt.Model):
    def __init__(
            self,
            num_neurons=2048,
            num_in_out=128,
            num_layers=1,
    ):
        super().__init__()

        self.emb = nn.Embedding(256, num_in_out)

        self.layers = nn.ModuleList([Bar(
            working_dim=num_in_out,
            num_neurons=num_neurons,
        ) for _ in range(num_layers)])

        self.dec = nn.Linear(num_in_out, 256)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dec(x)

        return x
