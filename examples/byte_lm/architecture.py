import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn


class ResidualSpike(snn.TTModule):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lif = snn.BSRLIF(
            hidden_dim,
            alpha=torch.rand(hidden_dim),
            beta=torch.rand(hidden_dim),
            gamma=torch.rand(hidden_dim),
            pos_threshold=torch.rand(hidden_dim),
            neg_threshold=torch.rand(hidden_dim),
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.lin.weight, 0.0, 0.01)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        spk = self.lif(x)
        delta = self.lin(spk)
        return x + delta


class SNNLM(snn.TTModule):
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
        # nn.init.xavier_uniform_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        emb_byte = self.emb(x)
        att_byte = self.net(emb_byte)
        pred_byte = self.dec(att_byte)

        return pred_byte
