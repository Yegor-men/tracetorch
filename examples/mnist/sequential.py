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
batch_size = 100
learning_rate = 1e-3

kernel_size = 4
stride = 4


# ============================================================================
# DATASET
# ============================================================================
class SequentialMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, pixel_permutation=None):
        self.ds = datasets.MNIST(root='../data', train=train, download=True, transform=transforms.ToTensor())
        if pixel_permutation is None:
            self.pixel_permutation = torch.randperm(784)
        else:
            self.pixel_permutation = pixel_permutation

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        C, H, W = img.shape
        img_flat = img.view(-1)
        img_shuffled = img_flat[self.pixel_permutation]
        img = img_shuffled.view(C, H, W)
        return img, label


def patch_collate(batch):
    imgs, labels = zip(*batch)
    imgs_b = torch.stack(imgs, dim=0)
    B, C, H, W = imgs_b.shape

    patches = imgs_b.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    n_h = patches.size(1)
    n_w = patches.size(2)
    n_patches = n_h * n_w

    patches = patches.view(B, n_patches, C * kernel_size * kernel_size)
    seq = patches.permute(1, 0, 2).contiguous()
    return seq, torch.tensor(labels, dtype=torch.long)


shared_permutation = torch.randperm(784)
train_dataset = SequentialMNIST(train=True, pixel_permutation=shared_permutation)
test_dataset = SequentialMNIST(train=False, pixel_permutation=shared_permutation)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=patch_collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=patch_collate)


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
class SeqSNN(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(kernel_size ** 2, 128),
            tt.snn.RLIB(128, beta=torch.rand(128), gamma=torch.rand(128), threshold=torch.rand(128)),
            nn.Linear(128, 128),
            tt.snn.RLIB(128, beta=torch.rand(128), gamma=torch.rand(128), threshold=torch.rand(128)),
            nn.Linear(128, 10)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.net(x)


class SeqRNN(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(kernel_size ** 2, 128),
            tt.rnn.GRU(128, 128, dim=-1),
            tt.rnn.GRU(128, 128, dim=-1),
            nn.Linear(128, 10)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.net(x)


class SeqSSM(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(kernel_size ** 2, 128),
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
        for seq, label in pbar:
            seq, label = seq.to(device), label.to(device)

            model.zero_grad()
            model.zero_states()

            final_output = None
            for t in range(seq.size(0)):
                final_output = model(seq[t])

            loss = loss_fn(final_output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = final_output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100. * correct / total:.2f}%"})

        train_losses.append(total_loss / len(train_dataloader))
        train_accs.append(100. * correct / total)

        model.eval()
        correct, total = 0, 0
        total_eval_loss = 0.0
        with torch.no_grad():
            for seq, label in test_dataloader:
                seq, label = seq.to(device), label.to(device)
                model.zero_states()
                final_output = None
                for t in range(seq.size(0)):
                    final_output = model(seq[t])

                loss = loss_fn(final_output, label)
                total_eval_loss += loss.item()
                pred = final_output.argmax(dim=1)
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
    models = {"SNN (RLIB)": SeqSNN().to(device), "RNN (GRU)": SeqRNN().to(device), "SSM (S6)": SeqSSM().to(device)}
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

    plt.suptitle('Sequential MNIST Classification (traceTorch)')
    plt.tight_layout()
    plt.show()
