import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn


def foobar(x):
    return nn.functional.sigmoid(4.0 * x)


dsrlits_min_timescale = tt.functional.halflife_to_decay(1)
dsrlits_max_timescale = tt.functional.halflife_to_decay(50)
dsrlits_diff = dsrlits_max_timescale - dsrlits_min_timescale
dsli_timescale = tt.functional.halflife_to_decay(150)


class ResidualSpike(snn.TTModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lif = snn.DSRLITS(
            hidden_dim,
            pos_alpha=torch.rand(hidden_dim),
            neg_alpha=torch.rand(hidden_dim),
            pos_beta=torch.rand(hidden_dim),
            neg_beta=torch.rand(hidden_dim),
            pos_gamma=torch.rand(hidden_dim),
            neg_gamma=torch.rand(hidden_dim),
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
    def __init__(self, in_features=256, hidden_features=1024, num_layers=10, latent_dim=4096):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_features, hidden_features), nn.Dropout(0.0))

        # Core SNN Body
        self.net = nn.Sequential(*[ResidualSpike(hidden_features) for _ in range(num_layers)])

        # Projection to massive latent state if dimensions differ
        self.proj = nn.Linear(hidden_features, latent_dim) if hidden_features != latent_dim else nn.Identity()

        # Terminal DSLI: This layer acts strictly as the tracked memory/latent-state
        self.state_layer = snn.DSLI(
            latent_dim,
            pos_alpha=torch.rand(latent_dim),
            neg_alpha=torch.rand(latent_dim),
            pos_beta=torch.rand(latent_dim),
            neg_beta=torch.rand(latent_dim),
        )

    def forward(self, x):
        features = self.net(self.enc(x))
        projected = self.proj(features)
        # Returns the massive hidden state vector directly
        return self.state_layer(projected)


class PredictiveDecoder(nn.Module):
    def __init__(self, latent_dim=4096, out_features=256):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(latent_dim, out_features)
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        return self.dec(x)


class ClassificationDecoder(nn.Module):
    def __init__(self, latent_dim=4096, num_classes=234):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(latent_dim, num_classes)
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        return self.dec(x)
