import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


class SimpleSNN(tt.Model):
    def __init__(self, c, n_labels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(c, 16, 3),
            snn.LIB(16, 0.9, 1.0, dim=-3),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 16),  # hardcoded for 10x10 input (10-3+1=8)
            snn.RLIB(16, 0.9, 0.9, 1.0),
            nn.Linear(16, n_labels),
            snn.LI(n_labels, 0.9),
        )

    def forward(self, x):
        return self.mlp(x)


def get_model_output(model, input_data, num_timesteps=10):
    model.eval()
    model.zero_states()
    outputs = []
    with torch.no_grad():
        for step in range(num_timesteps):
            output = model(input_data)
            outputs.append(output.detach().clone())
    return outputs


def compare_outputs(outputs1, outputs2, tolerance=1e-5):
    assert len(outputs1) == len(outputs2)
    for out1, out2 in zip(outputs1, outputs2):
        assert torch.allclose(out1, out2, atol=tolerance)


def test_compile_decompile():
    b, c, h, w = 2, 3, 10, 10
    model = SimpleSNN(c, b).to(device)
    random_feature = torch.rand(b, c, h, w).to(device)

    # 1. Baseline
    baseline_outputs = get_model_output(model, random_feature)

    # 2. Compile
    model.TTcompile()
    compiled_outputs = get_model_output(model, random_feature)
    compare_outputs(baseline_outputs, compiled_outputs)

    # 3. Decompile
    model.TTdecompile()
    decompiled_outputs = get_model_output(model, random_feature)
    compare_outputs(baseline_outputs, decompiled_outputs)


def test_save_load_states():
    b, c, h, w = 2, 3, 10, 10
    model = SimpleSNN(c, b).to(device)
    random_feature = torch.rand(b, c, h, w).to(device)

    model.eval()
    model.zero_states()
    for _ in range(5):
        _ = model(random_feature)

    states = model.save_states()
    assert len(states) > 0

    model2 = SimpleSNN(c, b).to(device)
    model2.load_state_dict(model.state_dict())
    model2.load_states(states, strict=False, device=device)

    # Test that outputs match exactly for the next timestep
    out1 = model(random_feature)
    out2 = model2(random_feature)
    assert torch.allclose(out1, out2)


if __name__ == "__main__":
    test_compile_decompile()
    test_save_load_states()
    print("All tests passed!")
