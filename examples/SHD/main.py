from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import tonic
import tonic.transforms as transforms
import tracetorch as tt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

num_epochs = 5
batch_size = 64
learning_rate = 1e-3

data_root = Path(__file__).resolve().parents[1] / "data"
sensor_size = tonic.datasets.SHD.sensor_size
num_inputs = sensor_size[0]
num_classes = 20


def get_dataloaders():
    frame_transform = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size, time_window=10000),
    ])

    train_dataset = tonic.datasets.SHD(save_to=data_root, train=True, transform=frame_transform)
    test_dataset = tonic.datasets.SHD(save_to=data_root, train=False, transform=frame_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        pin_memory=device == "cuda",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        pin_memory=device == "cuda",
    )
    return train_dataloader, test_dataloader


def flatten_events(events):
    while events.ndim > 3 and events.shape[-1] == 1:
        events = events.squeeze(-1)
    return events.float()


class SHDSNN(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_inputs, 256),
            tt.snn.LIB(256),
            nn.Linear(256, 256),
            tt.snn.LIB(256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SHDRNN(tt.Model):
    def __init__(self, cell_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_inputs, 256),
            cell_cls(256, 256),
            cell_cls(256, 256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_models():
    return {
        "SNN Identity": SHDSNN().to(device),
        "GRU": SHDRNN(tt.rnn.GRU).to(device),
        "LSTM": SHDRNN(tt.rnn.LSTM).to(device),
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
        for events, labels in dataloader:
            events = flatten_events(events).to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = forward_sequence(model, events)
            loss = F.cross_entropy(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, 100.0 * total_correct / total_count


def plot_history(history, epoch):
    plt.figure("SHD training curves", figsize=(12, 5))
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


def plot_shd_input(events, label):
    sample = events[:, 0].detach().cpu().T
    plt.figure("SHD diagnostic input", figsize=(10, 5))
    plt.imshow(sample, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
    plt.colorbar(label="event count")
    plt.xlabel("Timestep")
    plt.ylabel("Input neuron")
    plt.title(f"SHD framed event input, label={label.item()}")
    plt.tight_layout()
    plt.show(block=False)


def plot_model_outputs(models, events, label):
    fig, axes = plt.subplots(1, len(models), figsize=(3 * len(models), 3), sharey=True)
    if len(models) == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, (name, model) in zip(axes, models.items()):
            model.eval()
            logits = forward_sequence(model, events)
            probs = logits.softmax(dim=-1)[0].detach().cpu()
            prediction = int(probs.argmax())
            ax.bar(torch.arange(num_classes), probs)
            ax.set_xticks(range(0, num_classes, 2))
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{name}\npred={prediction}, label={label.item()}")

    fig.suptitle("Final timestep model outputs")
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

    events, labels = next(iter(test_dataloader))
    events = flatten_events(events).to(device)
    labels = labels.to(device)
    plot_shd_input(events, labels[0])
    plot_model_outputs(models, events, labels[0])
    plt.show()


if __name__ == "__main__":
    main()
