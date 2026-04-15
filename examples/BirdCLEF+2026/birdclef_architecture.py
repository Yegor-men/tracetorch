import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

min_timescale = tt.functional.halflife_to_decay(1)
max_timescale = tt.functional.halflife_to_decay(62.5)
timescale_diff = max_timescale - min_timescale


class ResidualSpike(tt.Model):
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
            quant_fn=nn.Identity(),
        )
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        # nn.init.normal_(self.lin2.weight, 0.0, 0.01)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x):
        return x + self.lin2(self.lif(self.lin1(x)))
        # return x + self.lin2(self.lif(x))


class BirdClassifierSNN(tt.Model):
    # Updated default in_features to 768 (256 Base + 256 Delta + 256 Delta-Delta)
    def __init__(self, in_features=768, hidden_features=1024, n1_layers=5, n2_layers=5, num_classes=234):
        super().__init__()
        self.enc = nn.Linear(in_features, hidden_features)
        nn.init.zeros_(self.enc.weight)
        nn.init.zeros_(self.enc.bias)

        self.n1 = nn.Sequential(*[ResidualSpike(hidden_features) for _ in range(n1_layers)])

        self.n2 = nn.Sequential(*[
            tt.ssm.Mamba(
                num_neurons=hidden_features,
                d_state=16,
            ) for _ in range(n2_layers)])

        self.dec = nn.Linear(hidden_features, num_classes)
        nn.init.zeros_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        return self.dec(self.n2(self.n1(self.enc(x))))
