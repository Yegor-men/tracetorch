"""Remember a random value through seven timesteps of silence."""

import torch
from torch import nn
import tracetorch as tt


class MemoryModel(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            tt.snn.LI(8),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    inputs = torch.zeros(8, 16, 1)
    inputs[0] = torch.randn(16, 1)
    targets = inputs[0].clone()
    model = MemoryModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    for update in range(300):
        # Each repetition starts a new sequence, so it must not inherit old memory.
        model.reset_states()
        optimizer.zero_grad(set_to_none=True)
        for timestep in inputs:
            prediction = model(timestep)
        loss = nn.functional.mse_loss(prediction, targets)
        if update == 0:
            print(f"Before training: {loss.item():.6f}")
        loss.backward()
        optimizer.step()

    model.eval()
    model.reset_states()
    with torch.no_grad():
        for timestep in inputs:
            prediction = model(timestep)
        loss = nn.functional.mse_loss(prediction, targets)
    print(f"After training:  {loss.item():.6f}")
    print("First inputs:    ", targets[:5, 0].tolist())
    print("Final predictions:", prediction[:5, 0].tolist())


if __name__ == "__main__":
    main()
