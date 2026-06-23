from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import tracetorch as tt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

num_epochs = 5
num_timesteps = 10
batch_size = 100
learning_rate = 1e-3

data_root = Path(__file__).resolve().parents[1] / "data"
num_classes = 10


def get_dataloaders():
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device == "cuda",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device == "cuda",
    )
    return train_dataloader, test_dataloader


def make_corrupted_sequence(images, alpha=None):
    batch_size_ = images.size(0)
    if alpha is None:
        alpha = torch.rand(batch_size_, 1, 1, 1, device=images.device)
    elif not isinstance(alpha, torch.Tensor):
        alpha = torch.full((batch_size_, 1, 1, 1), float(alpha), device=images.device)
    else:
        alpha = alpha.to(device=images.device, dtype=images.dtype)
        if alpha.ndim == 0:
            alpha = alpha.expand(batch_size_).view(batch_size_, 1, 1, 1)
        elif alpha.ndim == 1:
            alpha = alpha.view(batch_size_, 1, 1, 1)

    clean = images.unsqueeze(0).expand(num_timesteps, -1, -1, -1, -1)
    noise = torch.rand_like(clean)
    sequence = alpha.unsqueeze(0) * noise + (1.0 - alpha).unsqueeze(0) * clean

    return sequence, alpha.flatten()


def weighted_final_loss(logits, labels, alpha):
    loss = F.cross_entropy(logits, labels, reduction="none")
    weights = 1.0 / alpha.clamp_min(1e-4)
    weights = weights.clamp_max(100.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return (loss * weights).mean()


class ConvSNN(tt.Model):
    def __init__(self, quant_factory):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            tt.snn.LIB(32, dim=-3, quant_fn=quant_factory()),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            tt.snn.LIB(64, dim=-3, quant_fn=quant_factory()),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ConvRNN(tt.Model):
    def __init__(self, cell_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            cell_cls(32, 32, dim=-3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            cell_cls(64, 64, dim=-3),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_models():
    return {
        "SNN Identity": ConvSNN(lambda: nn.Identity()).to(device),
        "SNN Round": ConvSNN(lambda: tt.functional.round_ste()).to(device),
        "SNN Stochastic": ConvSNN(lambda: tt.functional.stochastic_round_ste()).to(device),
        "GRU": ConvRNN(tt.rnn.GRU).to(device),
        "LSTM": ConvRNN(tt.rnn.LSTM).to(device),
    }


def forward_sequence(model, sequence):
    model.zero_states()
    output = None
    for t in range(sequence.size(0)):
        output = model(sequence[t])
    return output


def run_epoch(model, dataloader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            sequence, alpha = make_corrupted_sequence(images)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = forward_sequence(model, sequence)
            loss = weighted_final_loss(logits, labels, alpha)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, 100.0 * total_correct / total_count


def plot_history(history, epoch):
    plt.figure("MNIST training curves", figsize=(12, 5))
    plt.clf()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.subplot(1, 2, 1)
    for idx, (name, values) in enumerate(history.items()):
        color = colors[idx % len(colors)]
        plt.plot(values["train_loss"], "--", color=color, label=f"{name} Train")
        plt.plot(values["eval_loss"], "-", color=color, linewidth=2, label=f"{name} Eval")
    plt.title(f"Loss after epoch {epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(fontsize="small")

    plt.subplot(1, 2, 2)
    for idx, (name, values) in enumerate(history.items()):
        color = colors[idx % len(colors)]
        plt.plot(values["train_acc"], "--", color=color, label=f"{name} Train")
        plt.plot(values["eval_acc"], "-", color=color, linewidth=2, label=f"{name} Eval")
    plt.title(f"Accuracy after epoch {epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend(fontsize="small")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def plot_mnist_sequence(sequence, label, alpha):
    frames = sequence[:, 0, 0].detach().cpu()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(frames[idx], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"t={idx}")
        ax.axis("off")
    fig.suptitle(f"Corrupted MNIST input, label={label.item()}, noise alpha={alpha[0].item():.3f}")
    plt.tight_layout()
    plt.show(block=False)


def plot_model_outputs(models, sequence, label):
    fig, axes = plt.subplots(1, len(models), figsize=(3 * len(models), 3), sharey=True)
    if len(models) == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, (name, model) in zip(axes, models.items()):
            model.eval()
            logits = forward_sequence(model, sequence)
            probs = logits.softmax(dim=-1)[0].detach().cpu()
            prediction = int(probs.argmax())
            ax.bar(torch.arange(num_classes), probs)
            ax.set_xticks(range(num_classes))
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{name}\npred={prediction}, label={label.item()}")

    fig.suptitle("Final timestep model outputs")
    plt.tight_layout()
    plt.show(block=False)


def evaluate_alpha_sweep(models, dataloader, alpha_values):
    sweep = {
        name: {"loss": [], "acc": []}
        for name in models
    }

    for alpha_value in tqdm(alpha_values.tolist(), desc="alpha sweep"):
        totals = {
            name: {"loss": 0.0, "correct": 0, "count": 0}
            for name in models
        }

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                sequence, _ = make_corrupted_sequence(images, alpha_value)

                for name, model in models.items():
                    model.eval()
                    logits = forward_sequence(model, sequence)
                    totals[name]["loss"] += F.cross_entropy(logits, labels, reduction="sum").item()
                    totals[name]["correct"] += (logits.argmax(dim=-1) == labels).sum().item()
                    totals[name]["count"] += labels.size(0)

        for name, values in totals.items():
            sweep[name]["loss"].append(values["loss"] / values["count"])
            sweep[name]["acc"].append(100.0 * values["correct"] / values["count"])

    return sweep


def plot_alpha_sweep(sweep, alpha_values):
    alpha_values = alpha_values.detach().cpu()
    plt.figure("MNIST alpha sweep", figsize=(12, 5))
    plt.clf()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.subplot(1, 2, 1)
    for idx, (name, values) in enumerate(sweep.items()):
        color = colors[idx % len(colors)]
        plt.plot(alpha_values, values["loss"], color=color, linewidth=2, label=name)
    plt.title("Loss by noise alpha")
    plt.xlabel("Noise alpha")
    plt.ylabel("Cross entropy")
    plt.grid(True)
    plt.legend(fontsize="small")

    plt.subplot(1, 2, 2)
    for idx, (name, values) in enumerate(sweep.items()):
        color = colors[idx % len(colors)]
        plt.plot(alpha_values, values["acc"], color=color, linewidth=2, label=name)
    plt.title("Accuracy by noise alpha")
    plt.xlabel("Noise alpha")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend(fontsize="small")

    plt.tight_layout()
    plt.show(block=False)


def main():
    train_dataloader, test_dataloader = get_dataloaders()
    models = build_models()
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=learning_rate)
        for name, model in models.items()
    }
    history = {
        name: {"train_loss": [], "train_acc": [], "eval_loss": [], "eval_acc": []}
        for name in models
    }

    for epoch in range(1, num_epochs + 1):
        for name, model in models.items():
            print(f"\nEpoch {epoch}/{num_epochs} - {name} ({sum(p.numel() for p in model.parameters()):,} params)")
            train_loss, train_acc = run_epoch(
                model,
                tqdm(train_dataloader, desc=f"{name} train"),
                optimizers[name],
            )
            eval_loss, eval_acc = run_epoch(
                model,
                tqdm(test_dataloader, desc=f"{name} eval"),
            )

            history[name]["train_loss"].append(train_loss)
            history[name]["train_acc"].append(train_acc)
            history[name]["eval_loss"].append(eval_loss)
            history[name]["eval_acc"].append(eval_acc)

            print(
                f"{name}: train loss={train_loss:.4f}, train acc={train_acc:.2f}% | "
                f"eval loss={eval_loss:.4f}, eval acc={eval_acc:.2f}%"
            )

        plot_history(history, epoch)

    images, labels = next(iter(test_dataloader))
    images = images.to(device)
    labels = labels.to(device)
    sequence, alpha = make_corrupted_sequence(images)
    plot_mnist_sequence(sequence, labels[0], alpha)
    plot_model_outputs(models, sequence, labels[0])

    alpha_values = torch.linspace(0.05, 0.95, 20)
    sweep = evaluate_alpha_sweep(models, test_dataloader, alpha_values)
    plot_alpha_sweep(sweep, alpha_values)
    plt.show()


if __name__ == "__main__":
    main()
