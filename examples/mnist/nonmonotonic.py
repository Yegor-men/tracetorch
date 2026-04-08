import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
import matplotlib.pyplot as plt
import tracetorch as tt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# ======================================================================================================================

min_prob, max_prob, noise_offset = 0.0, 1.0, 0.0
batch_size = 100
kernel_size = 4
stride = 4
pad = False
num_workers = 0
pin_memory = True


class MNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, min_val=0.0, max_val=1.0, offset=0.0, root="data"):
        self.ds = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * (max_val - min_val) + min_val + offset)
            ])
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


def patch_collate(batch):
    imgs, labels = zip(*batch)
    imgs_b = torch.stack(imgs, dim=0)
    B, C, H, W = imgs_b.shape

    if pad:
        rem_h = (H - kernel_size) % stride
        rem_w = (W - kernel_size) % stride
        pad_h = (stride - rem_h) % stride
        pad_w = (stride - rem_w) % stride
        if pad_h != 0 or pad_w != 0:
            imgs_b = nn.functional.pad(imgs_b, (0, pad_w, 0, pad_h), value=0.0)
            _, _, H, W = imgs_b.shape

    patches = imgs_b.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    n_h = patches.size(1)
    n_w = patches.size(2)
    n_patches = n_h * n_w
    patches = patches.view(B, n_patches, C * kernel_size * kernel_size)
    seq = patches.permute(1, 0, 2).contiguous()

    return imgs_b, seq, torch.tensor(labels, dtype=torch.long)


train_dataset = MNIST(train=True, min_val=min_prob, max_val=max_prob, offset=noise_offset)
test_dataset = MNIST(train=False, min_val=min_prob, max_val=max_prob, offset=noise_offset)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=patch_collate, num_workers=num_workers, pin_memory=pin_memory)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=patch_collate, num_workers=num_workers, pin_memory=pin_memory)

# ======================================================================================================================

min_scale = tt.functional.halflife_to_decay(16 / (kernel_size ** 2))
max_scale = tt.functional.halflife_to_decay(784 / (kernel_size ** 2))
scale_diff = max_scale - min_scale


class ParallelDLIEMA(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()

        # Initialize within your exact scale bounds
        pos_b = torch.rand(num_features) * scale_diff + min_scale
        neg_b = torch.rand(num_features) * scale_diff + min_scale

        # Use inverse sigmoid (logit) to store raw params
        self.raw_pos_beta = nn.Parameter(torch.logit(pos_b))
        self.raw_neg_beta = nn.Parameter(torch.logit(neg_b))

    def forward(self, x):
        """Processes the ENTIRE sequence tensor [T, B, E] at once."""
        T, B, E = x.shape

        pos_beta = torch.sigmoid(self.raw_pos_beta)
        neg_beta = torch.sigmoid(self.raw_neg_beta)

        # Apply the gated input logic exactly like the original DLIEMA
        x_pos = torch.where(x >= 0, x, 0.0) * (1 - pos_beta)
        x_neg = torch.where(x <= 0, x, 0.0) * (1 - neg_beta)

        # 1. CREATE CAUSAL CONVOLUTION KERNELS
        # We need decay powers:[beta^(T-1), beta^(T-2), ..., beta^1, 1]
        t = torch.arange(T - 1, -1, -1, device=x.device, dtype=x.dtype)

        k_pos = pos_beta.unsqueeze(1) ** t.unsqueeze(0)  # Shape: [E, T]
        k_neg = neg_beta.unsqueeze(1) ** t.unsqueeze(0)  # Shape: [E, T]

        # 2. PREPARE TENSORS FOR PyTorch Conv1D
        # PyTorch Conv1d expects [Batch, Channels, Length]
        x_pos_reshaped = x_pos.permute(1, 2, 0)
        x_neg_reshaped = x_neg.permute(1, 2, 0)

        # Causal Padding: Pad the sequence on the left by T - 1 so the filter never looks into the future
        x_pos_padded = nn.functional.pad(x_pos_reshaped, (T - 1, 0))
        x_neg_padded = nn.functional.pad(x_neg_reshaped, (T - 1, 0))

        # 3. FAST PARALLEL PROCESSING
        # groups=E applies a separate 1D filter to each channel/neuron natively in C++
        out_pos = nn.functional.conv1d(x_pos_padded, k_pos.unsqueeze(1), groups=E)
        out_neg = nn.functional.conv1d(x_neg_padded, k_neg.unsqueeze(1), groups=E)

        # Sum both paths
        mem = out_pos + out_neg

        # Return to [T, B, E]
        return mem.permute(2, 0, 1)


class ParallelDynamicLayer(nn.Module):
    def __init__(self, num_features: int, expansion: int = 1):
        super().__init__()
        dim = int(expansion * num_features)

        self.gate = nn.Linear(dim, dim)
        self.in_proj = nn.Linear(num_features, dim)
        self.lif = ParallelDLIEMA(dim)
        self.out_proj = nn.Linear(dim, num_features)

    def forward(self, x):
        # x is[T, B, E] -> Linear layers automatically broadcast over T and B!
        proj = self.in_proj(x)
        gate = nn.functional.silu(self.gate(proj))

        # Processes all timesteps in parallel via conv1d
        state = self.lif(proj)

        out = x + self.out_proj(gate * state)
        return out


class ParallelTest(nn.Module):
    def __init__(self, hidden_dim, num_layers, expansion):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(kernel_size ** 2, hidden_dim), nn.Tanh())
        self.layers = nn.ModuleList([ParallelDynamicLayer(hidden_dim, expansion) for _ in range(num_layers)])
        self.dec = nn.Linear(hidden_dim, 10)
        nn.init.zeros_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        # x is[T, B, input_dim]
        x = self.enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dec(x)
        return x  # Returns [T, B, 10]


model = ParallelTest(hidden_dim=128, num_layers=10, expansion=1).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
loss_fn = nn.functional.cross_entropy

train_losses, train_accs = [], []
num_epochs = 15

for e in range(num_epochs):
    model.train()
    for (img, seq, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        seq, label = torch.bernoulli(seq).to(device), label.to(device)

        model.zero_grad()

        # =======================================================
        # MAGIC HAPPENS HERE: No sequential loop.
        # ENTIRE TENSOR is processed in one pass!
        # =======================================================
        timestep_outputs = model(seq)  # Shape: [T, B, 10]

        T_steps = timestep_outputs.size(0)
        weights = torch.arange(1, T_steps + 1, dtype=torch.float32, device=device) / T_steps

        # Calculate loss over the sequence (vectorized via stacked list)
        losses = torch.stack([loss_fn(timestep_outputs[t], label) for t in range(T_steps)])
        loss = (losses * weights).sum() / weights.sum()

        pred_classes = timestep_outputs[-1].argmax(dim=-1)
        frac_correct = (pred_classes == label).sum().item() / batch_size

        train_losses.append(loss.item())
        train_accs.append(frac_correct)

        loss.backward()
        optimizer.step()

    plt.title("LOSS")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.show()

    plt.title("ACC")
    plt.plot(train_accs, label="train")
    plt.legend()
    plt.show()

    model.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        for (img, seq, label) in tqdm(test_dataloader, total=len(test_dataloader), desc=f"TEST - E{e}"):
            seq, label = torch.bernoulli(seq).to(device), label.to(device)

            timestep_outputs = model(seq)

            # Use only the last timestep for test evaluation
            model_output = timestep_outputs[-1]

            loss = loss_fn(model_output, label)
            test_loss += loss.item()
            pred_classes = model_output.argmax(dim=-1)
            frac_correct = (pred_classes == label).sum().item() / batch_size
            test_acc += frac_correct

        print(f"TEST - Loss: {test_loss / len(test_dataloader):.4f} | Acc: {test_acc / len(test_dataloader):.4f}")

with torch.no_grad():
    img_batch, seq_batch, label_batch = next(iter(test_dataloader))
    img_batch, seq_batch, label_batch = img_batch.to(device), seq_batch.to(device), label_batch.to(device)

    for i in range(10):
        # We slice i:i+1 to maintain the Batch dimension (B=1)
        img, seq, label = img_batch[i], seq_batch[:, i:i + 1], label_batch[i:i + 1]
        seq_bernoulli = torch.bernoulli(seq)

        # ONE pass gets the whole trace
        model_outputs_full = model(seq_bernoulli)

        input_spike_train = []
        model_outputs = []
        losses = []
        running_loss = 0.0
        T_steps = seq_bernoulli.size(0)

        for t in range(T_steps):
            spk_input = seq_bernoulli[t]  # [1, area]
            input_spike_train.append(spk_input.squeeze(0))

            model_output = model_outputs_full[t].squeeze(0)  # [10]
            model_outputs.append(nn.functional.softmax(model_output, dim=-1))

            loss = loss_fn(model_outputs_full[t], label)
            running_loss += loss.item() / T_steps
            losses.append(loss.item())

        tt.plot.render_image(img.unsqueeze(0), title=f"Loss: {running_loss:.3f}")
        input_spike_tensor = torch.stack(input_spike_train).T
        tt.plot.spike_train([input_spike_tensor.T[t] for t in range(input_spike_tensor.T.size(0))], title="Input")
        tt.plot.spike_train(model_outputs, title="Model Output")
        plt.title("Loss over time")
        plt.plot(losses)
        plt.show()
