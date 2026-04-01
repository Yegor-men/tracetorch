import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

min_timescale = tt.functional.halflife_to_decay(1)
max_timescale = tt.functional.halflife_to_decay(62.5)
timescale_diff = max_timescale - min_timescale


class ResidualSpike(snn.TTModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lif = snn.DSRLITS(
            hidden_dim,
            pos_alpha=torch.rand(hidden_dim) * timescale_diff + min_timescale,
            neg_alpha=torch.rand(hidden_dim) * timescale_diff + min_timescale,
            pos_beta=torch.rand(hidden_dim) * timescale_diff + min_timescale,
            neg_beta=torch.rand(hidden_dim) * timescale_diff + min_timescale,
            pos_gamma=torch.rand(hidden_dim) * timescale_diff + min_timescale,
            neg_gamma=torch.rand(hidden_dim) * timescale_diff + min_timescale,
            pos_threshold=torch.rand(hidden_dim),
            neg_threshold=torch.rand(hidden_dim),
            pos_scale=torch.rand(hidden_dim),
            neg_scale=torch.rand(hidden_dim),
            pos_rec_weight=torch.randn(hidden_dim) * 0.1,
            neg_rec_weight=torch.randn(hidden_dim) * 0.1,
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.lin.weight, 0.0, 0.01)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return x + self.lin(self.lif(torch.tanh(x)))


class BirdClassifierSNN(snn.TTModel):
    # Updated default in_features to 768 (256 Base + 256 Delta + 256 Delta-Delta)
    def __init__(self, in_features=768, hidden_features=1024, num_layers=10, num_classes=234):
        super().__init__()
        self.enc = nn.Linear(in_features, hidden_features)
        nn.init.zeros_(self.enc.weight)
        nn.init.zeros_(self.enc.bias)

        self.net = nn.Sequential(*[ResidualSpike(hidden_features) for _ in range(num_layers)])

        self.dec = nn.Sequential(
            snn.DSLI(
                hidden_features,
                pos_alpha=torch.rand(hidden_features) * timescale_diff + min_timescale,
                neg_alpha=torch.rand(hidden_features) * timescale_diff + min_timescale,
                pos_beta=torch.rand(hidden_features) * timescale_diff + min_timescale,
                neg_beta=torch.rand(hidden_features) * timescale_diff + min_timescale,
            ),
            nn.Linear(hidden_features, num_classes),
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        return self.dec(self.net(self.enc(x)))
