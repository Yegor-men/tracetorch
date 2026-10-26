"""Train a neuron with several synaptic memories for each input feature."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import tracetorch as tt


EXAMPLE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXAMPLE_ROOT.parent / "data"


def normalize_mixture(logits):
    return torch.softmax(logits, dim=-1)


class MultiTraceLIB(tt.Layer):
    """Filter each feature at several timescales, then integrate and fire.

    Input and output: [..., num_features]. Synaptic state: [..., num_features, K].
    Reset uses the previous smooth firing output before membrane decay.
    """

    def __init__(self, num_features, trace_decays=(0.2, 0.5, 0.8, 0.95)):
        super().__init__()
        self.num_features = num_features
        self.num_traces = len(trace_decays)
        self.define_state("syn", (num_features, self.num_traces))
        self.define_state("mem", (num_features,))
        self.define_state("prev_output", (num_features,))
        decays = torch.tensor(trace_decays).expand(num_features, -1).clone()
        self.define_parameter("alpha", decays, initialization_fn=tt.inverse_fn.sigmoid,
                              activation_fn=torch.sigmoid)
        self.define_parameter("mixture", torch.full_like(decays, 1 / self.num_traces),
                              initialization_fn=torch.log, activation_fn=normalize_mixture)
        self.define_parameter("beta", torch.full((num_features,), 0.9),
                              initialization_fn=tt.inverse_fn.sigmoid, activation_fn=torch.sigmoid)
        self.define_parameter("threshold", torch.ones(num_features),
                              initialization_fn=tt.inverse_fn.softplus, activation_fn=F.softplus)

    def forward(self, x):
        self.zero_states(x)
        self.syn = self.alpha * self.syn + (1 - self.alpha) * x.unsqueeze(-1)
        current = (self.mixture * self.syn).sum(dim=-1)
        self.mem = self.beta * (self.mem - self.threshold * self.prev_output) + current
        self.prev_output = tt.snn.spike_fn.smooth(self.mem - self.threshold)
        return self.prev_output


class CustomPatchSNN(tt.Model):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, width), MultiTraceLIB(width),
            nn.Linear(width, 10), tt.snn.LI(10, beta=0.9, beta_rank=0, learn_beta=False),
        )

    def forward(self, patch):
        return self.net(patch)


def images_to_patch_sequence(images):
    return F.unfold(images, kernel_size=4, stride=4).permute(2, 0, 1).contiguous()


def get_dataloaders(data_root, batch_size, train_samples, test_samples, seed):
    train = datasets.MNIST(data_root, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(data_root, train=False, download=True, transform=transforms.ToTensor())
    generator = torch.Generator().manual_seed(seed)
    if train_samples:
        train = Subset(train, torch.randperm(len(train), generator=generator)[:train_samples].tolist())
    if test_samples:
        test = Subset(test, torch.randperm(len(test), generator=generator)[:test_samples].tolist())
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator),
            DataLoader(test, batch_size=batch_size))


def forward_sequence(model, sequence):
    model.reset_states()
    for patch in sequence:
        logits = model(patch)
    return logits


def run_epoch(model, dataloader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_count = 0
    with torch.set_grad_enabled(training):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = forward_sequence(model, images_to_patch_sequence(images))
            loss = F.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.numel()
            total_correct += (logits.argmax(-1) == labels).sum().item()
            total_count += labels.numel()
    model.reset_states()
    return total_loss / total_count, total_correct / total_count


def plot_synaptic_memories():
    layer = MultiTraceLIB(1)
    inputs = torch.cat([torch.zeros(5), torch.ones(5), torch.zeros(40)])
    traces, currents = [], []
    with torch.no_grad():
        for value in inputs:
            layer(value.reshape(1, 1))
            traces.append(layer.syn[0, 0].clone())
            currents.append((layer.mixture[0] * layer.syn[0, 0]).sum())
    traces, currents = torch.stack(traces), torch.stack(currents)
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, layout="constrained")
    axes[0].step(range(len(inputs)), inputs)
    axes[0].set_ylabel("Input")
    for index, decay in enumerate((0.2, 0.5, 0.8, 0.95)):
        axes[1].plot(traces[:, index], label=f"decay {decay}")
    axes[1].set_ylabel("Synaptic history")
    axes[1].legend(ncol=2)
    axes[2].plot(currents, color="#7c3aed")
    axes[2].set_ylabel("Mixed current")
    axes[2].set_xlabel("Timestep")
    return fig


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    axes[0].plot(epochs, history["train_loss"], label="Training")
    axes[0].plot(range(len(history["test_loss"])), history["test_loss"], label="Test")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].plot(range(len(history["test_accuracy"])), history["test_accuracy"])
    axes[1].set_ylabel("Test accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch (0 = before training)")
    axes[0].legend()
    return fig


def plot_mixture(layer):
    fig, axis = plt.subplots(figsize=(7, 3.5), layout="constrained")
    image = axis.imshow(layer.mixture.detach().cpu(), aspect="auto", vmin=0, vmax=1)
    axis.set_xlabel("Synaptic history")
    axis.set_ylabel("Neuron")
    fig.colorbar(image, ax=axis, label="Learned mixture weight")
    return fig


def restore_continuation(model, sequence, checkpoint):
    """Restore matrix and vector states, then reproduce the remaining outputs."""
    model.eval()
    model.reset_states()
    cursor = 20
    device = sequence.device
    with torch.no_grad():
        for patch in sequence[:cursor]:
            model(patch)
        torch.save({"states": model.save_states(), "cursor": cursor,
                    "sequence": sequence.cpu()}, checkpoint.parent / "states.pt")
        expected = torch.stack([model(patch) for patch in sequence[cursor:]])
        saved_model = torch.load(checkpoint, map_location=device, weights_only=True)
        saved = torch.load(checkpoint.parent / "states.pt", map_location=device, weights_only=True)
        restored = CustomPatchSNN(width=saved_model["width"]).to(device).eval()
        restored.load_state_dict(saved_model["weights"])
        restored.load_states(saved["states"])
        actual = torch.stack([restored(patch) for patch in saved["sequence"][saved["cursor"]:]])
    torch.testing.assert_close(actual, expected)
    model.reset_states()
    return (actual - expected).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=EXAMPLE_ROOT / "figures")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or min(args.train_samples, args.test_samples) < 0:
        parser.error("epochs and batch size must be positive; sample limits must be nonnegative")
    plt.style.use("default")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    train, test = get_dataloaders(args.data_root, args.batch_size, args.train_samples, args.test_samples, 0)
    model = CustomPatchSNN().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, accuracy = run_epoch(model, test, args.device)
    history = {"train_loss": [], "test_loss": [loss], "test_accuracy": [accuracy]}
    print(f"Before training: loss={loss:.4f}, accuracy={accuracy:.1%}")
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train, args.device, optimizer)
        loss, accuracy = run_epoch(model, test, args.device)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(loss)
        history["test_accuracy"].append(accuracy)
        print(f"Epoch {epoch}: test loss={loss:.4f}, accuracy={accuracy:.1%}")
    print(f"Training and evaluation: {perf_counter() - started:.1f}s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_synaptic_memories().savefig(args.output_dir / "memories.png", dpi=150)
    plot_history(history).savefig(args.output_dir / "learning.png", dpi=150)
    plot_mixture(model.net[1]).savefig(args.output_dir / "mixture.png", dpi=150)
    plt.close("all")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    checkpoint = EXAMPLE_ROOT / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": model.state_dict(), "optimizer": optimizer.state_dict(), "width": 64}, checkpoint)
    images, _ = next(iter(test))
    sequence = images_to_patch_sequence(images[:2].to(args.device))
    difference = restore_continuation(model, sequence, checkpoint)
    print(f"Restored suffix matches uninterrupted output; maximum difference={difference:.3g}")
    print(f"Figures: {args.output_dir}; weights: {checkpoint}")


if __name__ == "__main__":
    main()
