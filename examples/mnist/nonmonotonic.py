import torch
from torch import nn
import torch.nn.functional as F

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
# NEW: Proper selective SSM (Mamba-style) – this replaces your ParallelDLIEMA entirely
# ======================================================================================================================

class ParallelMambaSSM(nn.Module):
    """
    Stable vectorized Mamba-style selective SSM.
    - No Python loop over T.
    - Uses the ratio-of-cumsums trick (Heinsen / common pure-PyTorch method).
    - Fully parallel over time for forward + backward.
    - Added safeguards against explosion (clamping + small eps).
    """

    def __init__(self, num_features: int, d_state: int = 16):
        super().__init__()
        self.num_features = num_features
        self.d_state = d_state

        # Fixed A: different time constants (HiPPO-inspired)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(num_features, 1)
        self.A_log = nn.Parameter(torch.log(A))  # A = -exp(A_log) < 0

        # Input-dependent projections
        self.dt_proj = nn.Linear(num_features, num_features, bias=True)
        self.B_proj = nn.Linear(num_features, d_state, bias=True)
        self.C_proj = nn.Linear(num_features, d_state, bias=True)

        self.D = nn.Parameter(torch.zeros(num_features))  # skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, B, E] → [T, B, E]"""
        T, B, E = x.shape
        assert E == self.num_features

        # 1. Compute all parameters in parallel
        delta = F.softplus(self.dt_proj(x)) + 1e-4  # [T, B, E]  (small offset for stability)
        B_all = self.B_proj(x)  # [T, B, N]
        C_all = self.C_proj(x)  # [T, B, N]

        A = -torch.exp(self.A_log)  # [E, N]  negative & stable

        # 2. Discretize (broadcast)
        deltaA = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # [T, B, E, N]
        barA = torch.exp(deltaA)  # [T, B, E, N]   decay < 1

        # barB (practical discretization)
        barB = delta.unsqueeze(-1) * B_all.unsqueeze(2)  # [T, B, E, N]

        # 3. Parallel scan via ratio of two cumsums (the key trick)
        #    This is equivalent to the recurrence but computed vectorized

        # Compute cumulative product of barA (prefix decays)
        # We use cumprod on barA, but shift for "decay up to previous"
        barA_cumprod = torch.cumprod(barA, dim=0)  # [T, B, E, N]

        # Shifted version for "decay from start up to t-1"
        barA_cum = torch.cat([
            torch.ones((1, B, E, self.d_state), device=x.device, dtype=x.dtype),
            barA_cumprod[:-1]
        ], dim=0)  # [T, B, E, N]

        # Contribution at each step
        x_exp = x.unsqueeze(-1)  # [T, B, E, 1]
        contrib = barB * x_exp  # [T, B, E, N]

        # Scaled contributions: contrib[k] / barA_cum[k]
        scaled_contrib = contrib / (barA_cum + 1e-12)  # avoid div-by-zero

        # Cumulative sum of scaled contributions
        cum_scaled = torch.cumsum(scaled_contrib, dim=0)  # [T, B, E, N]

        # Final hidden state h_t = barA_cum[t] * cum_scaled[t]
        h = barA_cum * cum_scaled  # [T, B, E, N]

        # 4. Output projection y_t = C · h_t
        y = torch.einsum("tb en, tb n -> tb e", h, C_all)  # [T, B, E]

        # Add skip connection
        y = y + self.D * x

        # Final safeguard (optional but helps when loss explodes early in training)
        y = torch.clamp(y, min=-100.0, max=100.0)

        return y


# ======================================================================================================================
# Updated layer – now uses the proper Mamba SSM instead of your dual-decay EMA
# ======================================================================================================================

class ParallelDynamicLayer(nn.Module):
    def __init__(self, num_features: int, expansion: int = 1, d_state: int = 16):
        super().__init__()
        dim = int(expansion * num_features)

        self.gate = nn.Linear(dim, dim)
        self.in_proj = nn.Linear(num_features, dim)
        self.lif = ParallelMambaSSM(dim, d_state=d_state)  # ← this is the only change
        self.out_proj = nn.Linear(dim, num_features)

    def forward(self, x):
        proj = self.in_proj(x)
        gate = F.silu(self.gate(proj))

        # lif now runs the full selective SSM (A, Δ, B, C, everything)
        state = self.lif(proj)

        out = x + self.out_proj(gate * state)
        return out


# ======================================================================================================================
# Model stays almost identical
# ======================================================================================================================

class ParallelTest(nn.Module):
    def __init__(self, hidden_dim, num_layers, expansion, d_state=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(kernel_size ** 2, hidden_dim), nn.Tanh())
        self.layers = nn.ModuleList([
            ParallelDynamicLayer(hidden_dim, expansion, d_state=d_state)
            for _ in range(num_layers)
        ])
        self.dec = nn.Linear(hidden_dim, 10)
        nn.init.zeros_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        x = self.enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dec(x)
        return x


model = ParallelTest(hidden_dim=128, num_layers=10, expansion=1, d_state=16).to(device)

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
