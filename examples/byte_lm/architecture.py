import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn


class Foobar(nn.Module):
    def __init__(self, slope: float = 4.0):
        super().__init__()
        self.slope = nn.Parameter(torch.log(torch.expm1(torch.Tensor([slope]))))

    def forward(self, x):
        return nn.functional.sigmoid(nn.functional.softplus(self.slope) * x)


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
            spike_fn=Foobar(4),
            deterministic=False,
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.lin.bias)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.ffn[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x):
        spk = self.lif(x)
        delta = self.lin(spk)
        return x + self.ffn(x + delta)


class SNNLM(snn.TTModel):
    def __init__(
            self,
            hidden_dim: int,
            num_layers: int,
            emb_dropout: float = 0.1,
            dec_dropout: float = 0.1,
    ):
        super().__init__()

        self.emb = nn.Sequential(
            nn.Embedding(256, hidden_dim),
            nn.Dropout(emb_dropout),
        )

        layers = [ResidualSpike(hidden_dim) for _ in range(num_layers)]
        self.net = nn.Sequential(*layers)

        self.dec = nn.Sequential(
            nn.Dropout(dec_dropout),
            nn.Linear(hidden_dim, 256)
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        emb_byte = self.emb(x)
        att_byte = self.net(emb_byte)
        pred_byte = self.dec(att_byte)

        return pred_byte
