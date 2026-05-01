import torch
from torch import nn
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
import tracetorch as tt
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# ============================================================================
# SETTINGS & HYPERPARAMETERS
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

num_epochs = 5
batch_size = 64
learning_rate = 1e-3

# The Heidelberg Spiking Digits dataset has 700 input neurons and 20 classes
sensor_size = tonic.datasets.SHD.sensor_size
num_classes = 20

# ============================================================================
# DATASET AND DATALOADERS
# ============================================================================
frame_transform = transforms.Compose([
    transforms.ToFrame(sensor_size=sensor_size, time_window=10000)
])

train_dataset = tonic.datasets.SHD(save_to='../data', train=True, transform=frame_transform)
test_dataset = tonic.datasets.SHD(save_to='../data', train=False, transform=frame_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=tonic.collation.PadTensors(batch_first=False))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=tonic.collation.PadTensors(batch_first=False))


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class SHD_SNN(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(sensor_size[0], 256),
            tt.snn.LIB(256, beta=torch.rand(256), threshold=torch.rand(256)),
            nn.Linear(256, 256),
            tt.snn.LIB(256, beta=torch.rand(256), threshold=torch.rand(256)),
            nn.Linear(256, num_classes)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.net(x)


class SHD_RNN(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(sensor_size[0], 256),
            tt.rnn.GRU(in_features=256, out_features=256),
            tt.rnn.GRU(in_features=256, out_features=256),
            nn.Linear(256, num_classes),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.net(x)


class SHD_SSM(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(sensor_size[0], 256),
            tt.ssm.S6(num_neurons=256, d_state=16),
            tt.ssm.S6(num_neurons=256, d_state=16),
            nn.Linear(256, num_classes),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.net(x)


# ============================================================================
# TRAINING & EVAL LOOP
# ============================================================================
def train_and_eval(model, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.functional.cross_entropy
    print(f"\n--- Training {name} on {device}: {sum(p.numel() for p in model.parameters()):,} parameters ---")
    train_losses, train_accs, eval_losses, eval_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        total_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}/{num_epochs}")
        for events, label in pbar:
            events = events.squeeze(-1).squeeze(-1)
            events, label = events.to(device), label.to(device)
            seq_len = events.size(0)

            model.zero_grad()
            model.zero_states()

            for t in range(seq_len):
                output = model(events[t])

            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100. * correct / total:.2f}%"})

        train_losses.append(total_loss / len(train_dataloader))
        train_accs.append(100. * correct / total)

        model.eval()
        correct, total = 0, 0
        total_eval_loss = 0.0
        with torch.no_grad():
            for events, label in test_dataloader:
                events = events.squeeze(-1).squeeze(-1)
                events, label = events.to(device), label.to(device)
                seq_len = events.size(0)
                model.zero_states()
                for t in range(seq_len):
                    output = model(events[t])

                loss = loss_fn(output, label)
                total_eval_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        eval_losses.append(total_eval_loss / len(test_dataloader))
        eval_accs.append(100. * correct / total)
        time.sleep(0.1)  # Prevents tqdm print overlap
        print(
            f"Epoch {epoch + 1} | Train Loss: {train_losses[-1]:.4f} Acc: {train_accs[-1]:.2f}% | Eval Loss: {eval_losses[-1]:.4f} Acc: {eval_accs[-1]:.2f}%")
        time.sleep(0.2)

    return train_losses, train_accs, eval_losses, eval_accs


if __name__ == "__main__":
    models = {"SNN": SHD_SNN().to(device), "RNN (GRU)": SHD_RNN().to(device), "SSM (S6)": SHD_SSM().to(device)}
    results = {}

    for name, model in models.items():
        results[name] = train_and_eval(model, name)

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    plt.figure(figsize=(12, 5))

    # Loss Subplot
    plt.subplot(1, 2, 1)
    for i, (name, (t_loss, t_acc, e_loss, e_acc)) in enumerate(results.items()):
        color = colors[i % len(colors)]
        plt.plot(t_loss, label=f'{name} (Train)', linestyle='--', color=color)
        plt.plot(e_loss, label=f'{name} (Eval)', linestyle='-', linewidth=2, color=color)
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Subplot
    plt.subplot(1, 2, 2)
    for i, (name, (t_loss, t_acc, e_loss, e_acc)) in enumerate(results.items()):
        color = colors[i % len(colors)]
        plt.plot(t_acc, label=f'{name} (Train)', linestyle='--', color=color)
        plt.plot(e_acc, label=f'{name} (Eval)', linestyle='-', linewidth=2, color=color)
    plt.title('Accuracy over Epochs (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.suptitle('Heidelberg Digits Classification (traceTorch Models)')
    plt.tight_layout()
    plt.show()
