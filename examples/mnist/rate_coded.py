import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
num_timesteps = 20
batch_size = 100
learning_rate = 1e-3

# ============================================================================
# DATASET
# ============================================================================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class RateSNN(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            tt.snn.LIB(128, beta=torch.rand(128), threshold=torch.rand(128)),
            nn.Linear(128, 128),
            tt.snn.LIB(128, beta=torch.rand(128), threshold=torch.rand(128)),
            nn.Linear(128, 10)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.net(x)


class RateRNN(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            tt.rnn.GRU(128, 128, dim=-1),
            tt.rnn.GRU(128, 128, dim=-1),
            nn.Linear(128, 10)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.net(x)


class RateSSM(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            tt.ssm.S6(128, 16),
            tt.ssm.S6(128, 16),
            nn.Linear(128, 10)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.net(x)


# ============================================================================
# TRAINING LOOP
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
        for image, label in pbar:
            image, label = image.to(device), label.to(device)

            model.zero_grad()
            model.zero_states()

            running_loss = 0.0
            for t in range(num_timesteps):
                spk_image = torch.bernoulli(image)
                output = model(spk_image)
                loss = loss_fn(output, label)
                running_loss += loss

            running_loss = running_loss / num_timesteps
            running_loss.backward()
            optimizer.step()

            total_loss += running_loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            pbar.set_postfix({"Loss": f"{running_loss.item():.4f}", "Acc": f"{100. * correct / total:.2f}%"})

        train_losses.append(total_loss / len(train_dataloader))
        train_accs.append(100. * correct / total)

        model.eval()
        correct, total = 0, 0
        total_eval_loss = 0.0
        with torch.no_grad():
            for image, label in test_dataloader:
                image, label = image.to(device), label.to(device)
                model.zero_states()
                running_output = 0
                for t in range(num_timesteps):
                    spk_image = torch.bernoulli(image)
                    running_output += model(spk_image)
                avg_output = running_output / num_timesteps
                loss = loss_fn(avg_output, label)
                total_eval_loss += loss.item()
                pred = avg_output.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        eval_losses.append(total_eval_loss / len(test_dataloader))
        eval_accs.append(100. * correct / total)
        time.sleep(0.1)
        print(
            f"Epoch {epoch + 1} | Train Loss: {train_losses[-1]:.4f} Acc: {train_accs[-1]:.2f}% | Eval Loss: {eval_losses[-1]:.4f} Acc: {eval_accs[-1]:.2f}%")
        time.sleep(0.2)

    return train_losses, train_accs, eval_losses, eval_accs


if __name__ == "__main__":
    models = {"SNN (LIB)": RateSNN().to(device), "RNN (GRU)": RateRNN().to(device), "SSM (S6)": RateSSM().to(device)}
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

    plt.suptitle('Rate-Coded MNIST Classification (traceTorch)')
    plt.tight_layout()
    plt.show()
