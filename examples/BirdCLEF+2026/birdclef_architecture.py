import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn


def foobar(x):
    return nn.functional.sigmoid(4.0 * x)


dsrlits_min_timescale = tt.functional.halflife_to_decay(1)
dsrlits_max_timescale = tt.functional.halflife_to_decay(50)
dsrlits_diff = dsrlits_max_timescale - dsrlits_min_timescale


class ResidualSpike(snn.TTModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lif = snn.DSRLITS(
            hidden_dim,
            pos_alpha=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            neg_alpha=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            pos_beta=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            neg_beta=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            pos_gamma=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            neg_gamma=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            pos_threshold=torch.rand(hidden_dim),
            neg_threshold=torch.rand(hidden_dim),
            pos_scale=torch.randn(hidden_dim) * 0.5 + 1.0,
            neg_scale=torch.randn(hidden_dim) * 0.5 + 1.0,
            pos_rec_weight=torch.randn(hidden_dim) * 0.1,
            neg_rec_weight=torch.randn(hidden_dim) * 0.1,
            spike_fn=foobar,
            deterministic=False,
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return x + self.lin(self.lif(x))


class SNNWorldModel(snn.TTModel):
    def __init__(self, in_features=256, hidden_features=1024, num_layers=10):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_features, hidden_features), nn.Dropout(0.0))

        self.net = nn.Sequential(*[ResidualSpike(hidden_features) for _ in range(num_layers)])

    def forward(self, x):
        emb_sound = self.enc(x)
        att_sound = self.net(emb_sound)
        return att_sound


class PredictiveDecoder(nn.Module):
    def __init__(self, latent_dim=1024, out_features=256):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(latent_dim, out_features)
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        return self.dec(x)
2

class ClassificationDecoder(nn.Module):
    def __init__(self, latent_dim=1024, num_classes=234):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(latent_dim, num_classes)
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        return self.dec(x)
