"""Read MNIST images as 49 patches and train from the final output.

Run: python examples/mnist/main.py --device cpu
"""

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


def images_to_patch_sequence(images):
    """Convert [B, 1, 28, 28] images to [49, B, 16] patches in row order."""
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError("Expected MNIST images with shape [B, 1, 28, 28]")
    patches = F.unfold(images, kernel_size=4, stride=4)
    return patches.permute(2, 0, 1).contiguous()


class PatchSNN(tt.Model):
    """Process one [B, 16] patch and return ten scores for each image."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 256),
            tt.snn.LIB(256),
            nn.Linear(256, 256),
            tt.snn.LIB(256),
            nn.Linear(256, 10),
            tt.snn.LI(10, beta=0.9, beta_rank=0, learn_beta=False),
        )

    def forward(self, patch):
        return self.net(patch)


def get_dataloaders(data_root, batch_size, train_samples, test_samples, seed):
    transform = transforms.ToTensor()
    train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    if train_samples:
        indices = torch.randperm(len(train), generator=generator)[:train_samples].tolist()
        train = Subset(train, indices)
    if test_samples:
        indices = torch.randperm(len(test), generator=generator)[:test_samples].tolist()
        test = Subset(test, indices)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(test, batch_size=batch_size),
    )


def forward_sequence(model, sequence):
    # Patches from one image share memory. The next batch contains new images.
    model.reset_states()
    for patch in sequence:
        logits = model(patch)
    return logits


def train_batch(model, images, labels, optimizer):
    """Update weights once, using the final output of each image sequence."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sequence = images_to_patch_sequence(images)
    logits = forward_sequence(model, sequence)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.detach(), logits.detach()


def run_epoch(model, dataloader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_count = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        if training:
            loss, logits = train_batch(model, images, labels, optimizer)
        else:
            with torch.no_grad():
                sequence = images_to_patch_sequence(images)
                logits = forward_sequence(model, sequence)
                loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.numel()
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_count += labels.numel()
    model.reset_states()
    return total_loss / total_count, total_correct / total_count


def record_neuron_response(inputs, beta):
    """Record a fixed LIB neuron's membrane and firing after each input."""
    neuron = tt.snn.LIB(1, beta=beta, threshold=1.0, learn_beta=False, learn_threshold=False)
    membrane, spikes = [], []
    with torch.no_grad():
        for value in inputs:
            output = neuron(value.reshape(1, 1))
            membrane.append(neuron.mem.item())
            spikes.append(output.item())
    return torch.tensor(membrane), torch.tensor(spikes)


def record_readout(model, image, device):
    """Inspect all timestep scores without adding losses at earlier timesteps."""
    sequence = images_to_patch_sequence(image.unsqueeze(0).to(device))
    model.eval()
    model.reset_states()
    scores = []
    with torch.no_grad():
        for patch in sequence:
            scores.append(model(patch)[0].cpu())
    model.reset_states()
    return torch.stack(scores)


def plot_patch_sequence(image):
    patches = images_to_patch_sequence(image.unsqueeze(0))[:, 0]
    fig = plt.figure(figsize=(10, 5), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    original = fig.add_subplot(grid[0])
    original.imshow(image[0], cmap="gray", vmin=0, vmax=1)
    for boundary in range(4, 28, 4):
        original.axhline(boundary - 0.5, color="#f59e0b", linewidth=0.7)
        original.axvline(boundary - 0.5, color="#f59e0b", linewidth=0.7)
    original.set_title("One image, divided into 4 × 4 patches")
    original.axis("off")
    patch_grid = grid[1].subgridspec(7, 7, wspace=0.15, hspace=0.4)
    for index, patch in enumerate(patches):
        axis = fig.add_subplot(patch_grid[index // 7, index % 7])
        axis.imshow(patch.reshape(4, 4), cmap="gray", vmin=0, vmax=1)
        axis.set_title(str(index + 1), fontsize=8, pad=1)
        axis.axis("off")
    fig.suptitle("Patch order: left to right, then the next row")
    return fig


def plot_neuron_response():
    inputs = torch.cat([torch.zeros(5), torch.full((20,), 0.35), torch.zeros(15)])
    steps = torch.arange(1, len(inputs) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, layout="constrained")
    axes[0].step(steps, inputs, where="mid", color="#374151")
    axes[0].set_ylabel("Input")
    for index, beta in enumerate((0.5, 0.8, 0.95)):
        membrane, spikes = record_neuron_response(inputs, beta)
        line, = axes[1].plot(steps, membrane, label=f"decay = {beta}")
        firing_steps = steps[spikes.bool()]
        axes[2].scatter(firing_steps, torch.full_like(firing_steps, index),
                        marker="|", s=120, color=line.get_color())
    axes[1].axhline(1.0, color="#374151", linestyle="--", label="threshold")
    axes[1].set_ylabel("Membrane")
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    axes[2].set_yticks([0, 1, 2], ["0.5", "0.8", "0.95"])
    axes[2].set_ylabel("Decay\n(firing events)")
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylim(-0.5, 2.5)
    fig.suptitle("The same input, three membrane decays")
    return fig


def plot_firing_function():
    distance = torch.linspace(-1.5, 1.5, 301)
    hard = tt.snn.spike_fn.deterministic(distance)
    smooth = tt.snn.spike_fn.smooth(distance)
    derivative = 4 * smooth * (1 - smooth)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].plot(distance, hard, label="Forward output")
    axes[0].plot(distance, smooth, linestyle="--", label="Smooth function")
    axes[0].set_ylabel("Firing output")
    axes[0].legend()
    axes[1].plot(distance, derivative, color="#b45309")
    axes[1].set_ylabel("Derivative used in backward pass")
    for axis in axes:
        axis.set_xlabel("Membrane minus threshold")
        axis.axvline(0, color="#374151", linewidth=0.5)
        axis.grid(alpha=0.2)
    fig.suptitle("Hard forward output; smooth backward derivative")
    return fig


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(epochs, history["train_loss"], label="Training")
    axes[0].plot([0, *epochs], history["test_loss"], label="Test")
    axes[0].set_ylabel("Mean cross-entropy loss")
    axes[1].plot(epochs, [100 * value for value in history["train_accuracy"]], label="Training")
    axes[1].plot([0, *epochs], [100 * value for value in history["test_accuracy"]], label="Test")
    axes[1].set_ylabel("Correct predictions (%)")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.set_xlabel("Epoch (0 = before training)")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle("Learning from 49-patch image sequences")
    return fig


def plot_predictions(model, images, labels, device):
    model.eval()
    with torch.no_grad():
        logits = forward_sequence(model, images_to_patch_sequence(images.to(device)))
    predicted = logits.argmax(-1).cpu()
    model.reset_states()
    # Include errors so the figure shows what this short run has not learned.
    correct = (predicted == labels).nonzero().flatten()[:8]
    incorrect = (predicted != labels).nonzero().flatten()[:8]
    selected = torch.cat([correct, incorrect]).tolist()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8), layout="constrained")
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, selected):
        axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
        color = "#166534" if predicted[index] == labels[index] else "#b91c1c"
        axis.set_title(f"Label {labels[index].item()} / predicted {predicted[index].item()}",
                       color=color, fontsize=9)
    fig.suptitle("Test examples: correct predictions and errors")
    return fig


def plot_readout(model, image, label, device):
    probabilities = record_readout(model, image, device).softmax(-1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[1, 3], layout="constrained")
    axes[0].imshow(image[0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Label {label.item()}")
    axes[0].axis("off")
    for digit in range(10):
        axes[1].plot(range(1, 50), probabilities[:, digit], label=str(digit),
                     linewidth=2 if digit == label.item() else 1,
                     alpha=1 if digit == label.item() else 0.5)
    axes[1].set_xlabel("Patch timestep")
    axes[1].set_ylabel("Probability from readout scores")
    axes[1].set_ylim(0, 1)
    axes[1].legend(title="Digit", ncol=5, fontsize=8)
    axes[1].axvline(49, color="#374151", linestyle="--", linewidth=0.8)
    fig.suptitle("Readout during the sequence; prediction uses timestep 49")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=4096, help="0 uses the full training set")
    parser.add_argument("--test-samples", type=int, default=1024, help="0 uses the full test set")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=EXAMPLE_ROOT / "figures")
    parser.add_argument("--no-plots", action="store_true", help="Skip figures; still save training history")
    parser.add_argument("--show-plots", action="store_true", help="Also open the figures in a graphical window")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size, threads and learning rate must be positive")
    if min(args.train_samples, args.test_samples) < 0:
        parser.error("sample limits must be nonnegative")
    if args.no_plots and args.show_plots:
        parser.error("choose either --no-plots or --show-plots")
    return args


def main():
    args = parse_args()
    plt.style.use("default")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train, test = get_dataloaders(
        args.data_root, args.batch_size, args.train_samples, args.test_samples, args.seed,
    )
    model = PatchSNN().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"Sequential MNIST: {len(train.dataset)} train / {len(test.dataset)} test; {args.device}")
    baseline_loss, baseline_accuracy = run_epoch(model, test, args.device)
    print(f"Before training: test loss={baseline_loss:.4f}, accuracy={baseline_accuracy:.1%}")
    history = {"train_loss": [], "train_accuracy": [],
               "test_loss": [baseline_loss], "test_accuracy": [baseline_accuracy]}
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train, args.device, optimizer)
        test_loss, test_accuracy = run_epoch(model, test, args.device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        print(f"Epoch {epoch}: train loss={train_loss:.4f}, accuracy={train_accuracy:.1%}; "
              f"test loss={test_loss:.4f}, accuracy={test_accuracy:.1%}")
    elapsed = perf_counter() - started
    print(f"Training and evaluation: {elapsed:.1f}s (download and figures excluded)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"history": history, "training_seconds": elapsed,
              "train_samples": len(train.dataset), "test_samples": len(test.dataset),
              "settings": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()}}
    (args.output_dir / "history.json").write_text(json.dumps(result, indent=2) + "\n")
    checkpoint = EXAMPLE_ROOT / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                "completed_epochs": args.epochs}, checkpoint)
    print(f"Saved model: {checkpoint}")
    if not args.no_plots:
        images, labels = next(iter(test))
        figures = {
            "patches": plot_patch_sequence(images[0]),
            "neuron": plot_neuron_response(),
            "firing": plot_firing_function(),
            "learning": plot_history(history),
            "predictions": plot_predictions(model, images, labels, args.device),
            "readout": plot_readout(model, images[0], labels[0], args.device),
        }
        for name, figure in figures.items():
            figure.savefig(args.output_dir / f"{name}.png", dpi=150)
        if args.show_plots:
            plt.show()
        for figure in figures.values():
            plt.close(figure)
    print(f"Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()
