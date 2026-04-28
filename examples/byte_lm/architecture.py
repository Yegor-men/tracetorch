import torch
from torch import nn
import tracetorch as tt


class Residual(tt.Model):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = int(num_neurons)

        self.BD = nn.Linear(num_neurons, 2 * num_neurons, bias=False)
        self.C = nn.Linear(num_neurons, num_neurons, bias=False)
        # nn.init.zeros_(self.BD.weight)
        # nn.init.zeros_(self.C.weight)

        self.lif = tt.snn.RLIB(
            num_neurons,
            beta=torch.rand(num_neurons),
            gamma=torch.rand(num_neurons),
            threshold=torch.rand(num_neurons),
            quant_fn=nn.Identity(),
        )

    def forward(self, x):
        B, D = torch.split(self.BD(x), [self.num_neurons, self.num_neurons], dim=-1)
        C = self.C(self.lif(B))
        return x + C * nn.functional.silu(D)


class TTLM(tt.Model):
    def __init__(
            self,
            num_neurons=1024,
            num_layers=10,
    ):
        super().__init__()
        self.emb = nn.Embedding(256, num_neurons)

        self.layers = nn.ModuleList([
            Residual(
                num_neurons=num_neurons,
            ) for _ in range(num_layers)
        ])

        # self.layers = nn.ModuleList([
        #     tt.ssm.Mamba(
        #         num_neurons=num_neurons,
        #         d_state=16,
        #     ) for _ in range(num_layers)
        # ])

        # self.ssm = tt.ssm.Mamba(num_neurons, 16)

        self.dec = nn.Linear(num_neurons, 256, bias=False)
        nn.init.zeros_(self.dec.weight)

    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dec(x)

        return x
