"""Train a residual SNN on a bounded subset of Spiking Heidelberg Digits.

Run: python examples/shd/main.py --device cpu
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import tonic
from tonic import transforms

import tracetorch as tt


EXAMPLE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXAMPLE_ROOT.parent / "data"


def collate_left_padded(samples):
    """Align recording endings and return events [T, B, 700] and labels [B]."""
    recordings, labels = zip(*samples)
    frames = [torch.as_tensor(recording).flatten(start_dim=1).float() for recording in recordings]
    longest = max(frame.shape[0] for frame in frames)
    batch = frames[0].new_zeros(longest, len(frames), frames[0].shape[-1])
    for index, frame in enumerate(frames):
        batch[-len(frame):, index] = frame
    return batch, torch.as_tensor(labels, dtype=torch.long)


def sequence_decays(num_features, sequence_steps):
    """Initialize passive memory horizons between one and four recordings."""
    horizons = max(2, sequence_steps) * (1 + 3 * torch.rand(num_features))
    return tt.utils.convert_decay(horizons, "horizon", "decay")


class ResidualBlock(nn.Module):
    def __init__(self, width, sequence_steps):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(width, width),
            tt.snn.SLIB(
                width, alpha=torch.rand(width), beta=sequence_decays(width, sequence_steps),
                spike_fn=tt.snn.spike_fn.smooth,
            ),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return x + self.branch(x)


class ResidualSNN(tt.Model):
    def __init__(self, width=64, sequence_steps=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(700, width),
            ResidualBlock(width, sequence_steps),
            ResidualBlock(width, sequence_steps),
            tt.snn.LI(width, beta=sequence_decays(width, sequence_steps)),
            nn.Linear(width, 20),
        )
        # The feature accumulator can sum dozens of frames before this readout.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, events):
        return self.net(events)


def get_dataloaders(data_root, batch_size, train_samples, test_samples, time_window, seed):
    frame_transform = transforms.ToFrame(
        sensor_size=tonic.datasets.SHD.sensor_size, time_window=time_window,
    )
    train = tonic.datasets.SHD(data_root, train=True, transform=frame_transform)
    test = tonic.datasets.SHD(data_root, train=False, transform=frame_transform)
    generator = torch.Generator().manual_seed(seed)
    if train_samples:
        train = Subset(train, torch.randperm(len(train), generator=generator)[:train_samples].tolist())
    if test_samples:
        test = Subset(test, torch.randperm(len(test), generator=generator)[:test_samples].tolist())
    # Frame once: otherwise every epoch reopens HDF5 files and rebins the same events.
    train = [train[index] for index in range(len(train))]
    test = [test[index] for index in range(len(test))]
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator,
                   collate_fn=collate_left_padded),
        DataLoader(test, batch_size=batch_size, collate_fn=collate_left_padded),
    )


def forward_sequence(model, sequence):
    model.reset_states()
    for events in sequence:
        logits = model(events)
    return logits


def run_epoch(model, dataloader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_count = 0
    with torch.set_grad_enabled(training):
        for events, labels in dataloader:
            events, labels = events.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = forward_sequence(model, events)
            loss = F.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.numel()
            total_correct += (logits.argmax(-1) == labels).sum().item()
            total_count += labels.numel()
    model.reset_states()
    return total_loss / total_count, total_correct / total_count


def plot_recording(events, time_window):
    fig, axis = plt.subplots(figsize=(9, 3), layout="constrained")
    image = axis.imshow(events[:, 0].T, aspect="auto", origin="lower", cmap="magma")
    axis.set_xlabel(f"Frame ({time_window / 1000:g} ms each)")
    axis.set_ylabel("Input neuron")
    fig.colorbar(image, ax=axis, label="Event count")
    return fig


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    axes[0].plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Training")
    axes[0].plot(history["test_loss"], label="Test")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(history["test_accuracy"])
    axes[1].set_ylabel("Test accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch (0 = before training)")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--time-window", type=int, default=20000, help="Frame duration in microseconds")
    parser.add_argument("--train-samples", type=int, default=1024, help="0 uses the full training set")
    parser.add_argument("--test-samples", type=int, default=256, help="0 uses the full test set")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=EXAMPLE_ROOT / "figures")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.width, args.time_window, args.threads) <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size, width, time window, threads and learning rate must be positive")
    if min(args.train_samples, args.test_samples) < 0:
        parser.error("sample limits must be nonnegative")
    return args


def main():
    args = parse_args()
    plt.style.use("default")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    preparing = perf_counter()
    train, test = get_dataloaders(
        args.data_root, args.batch_size, args.train_samples, args.test_samples, args.time_window, args.seed,
    )
    print(f"Dataset download and framing: {perf_counter() - preparing:.1f}s")
    sequence_steps = max(recording.shape[0] for recording, _ in train.dataset)
    model = ResidualSNN(args.width, sequence_steps).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"SHD: {len(train.dataset)} train / {len(test.dataset)} test; {args.device}; "
          f"longest training recording: {sequence_steps} frames")
    baseline_loss, baseline_accuracy = run_epoch(model, test, args.device)
    print(f"Before training: test loss={baseline_loss:.4f}, accuracy={baseline_accuracy:.1%}")
    history = {"train_loss": [], "test_loss": [baseline_loss], "test_accuracy": [baseline_accuracy]}
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train, args.device, optimizer)
        model.eval()
        model.compile_parameters()
        try:
            test_loss, test_accuracy = run_epoch(model, test, args.device)
        finally:
            # The next epoch must use live, trainable parameters again.
            model.decompile_parameters()
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        print(f"Epoch {epoch}: train loss={train_loss:.4f}, accuracy={train_accuracy:.1%}; "
              f"test loss={test_loss:.4f}, accuracy={test_accuracy:.1%}")
    runtime = perf_counter() - started
    print(f"Training and evaluation: {runtime:.1f}s (preparation excluded)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events, _ = next(iter(test))
    plot_recording(events, args.time_window).savefig(args.output_dir / "recording.png", dpi=150)
    plot_history(history).savefig(args.output_dir / "learning.png", dpi=150)
    plt.close("all")
    (args.output_dir / "history.json").write_text(json.dumps(
        {**history, "runtime_seconds": runtime, "sequence_steps": sequence_steps}, indent=2,
    ))
    checkpoint = EXAMPLE_ROOT / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                "width": args.width, "sequence_steps": sequence_steps,
                "time_window": args.time_window}, checkpoint)
    print(f"Figures: {args.output_dir}; weights: {checkpoint}")


if __name__ == "__main__":
    main()
