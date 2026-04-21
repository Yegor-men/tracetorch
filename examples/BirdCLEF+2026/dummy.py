import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

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
    def __init__(self, train=True, min_val=0.0, max_val=1.0, offset=0.0, root="data",
                 pixel_permutation=None):
        # Use shared permutation or create one if this is the first instance
        if pixel_permutation is None:
            self.pixel_permutation = torch.randperm(784)
        else:
            self.pixel_permutation = pixel_permutation

        self.ds = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),  # -> [C, H, W] in [0,1]
                transforms.Lambda(lambda x: x * (max_val - min_val) + min_val + offset)
            ])
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]  # img: [C, H, W], label: int

        # Apply consistent pixel shuffling
        C, H, W = img.shape
        img_flat = img.view(-1)  # Flatten to [C*H*W]
        img_shuffled = img_flat[self.pixel_permutation]
        img = img_shuffled.view(C, H, W)  # Reshape back

        return img, label


def patch_collate(batch):
    """
    returns:
      imgs_orig:  [B, C, H, W]
      seq:        [T, B, area]  (area = C * k * k)
      labels_onehot: [B, 10]
    """
    imgs, labels = zip(*batch)
    imgs_b = torch.stack(imgs, dim=0)  # [B, C, H, W]
    B, C, H, W = imgs_b.shape

    # optional pad so patches tile exactly
    if pad:
        rem_h = (H - kernel_size) % stride
        rem_w = (W - kernel_size) % stride
        pad_h = (stride - rem_h) % stride
        pad_w = (stride - rem_w) % stride
        if pad_h != 0 or pad_w != 0:
            imgs_b = nn.functional.pad(imgs_b, (0, pad_w, 0, pad_h), value=0.0)  # pad (left,right,top,bottom) order
            _, _, H, W = imgs_b.shape

    # unfold to patches: [B, C, n_h, n_w, k, k]
    patches = imgs_b.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    # permute and reshape to [B, n_patches, area]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    n_h = patches.size(1)
    n_w = patches.size(2)
    n_patches = n_h * n_w
    patches = patches.view(B, n_patches, C * kernel_size * kernel_size)  # [B, T, area]
    # transpose to [T, B, area]
    seq = patches.permute(1, 0, 2).contiguous()  # [T, B, area]

    return imgs_b, seq, torch.tensor(labels, dtype=torch.long)


# Create shared pixel permutation
shared_permutation = torch.randperm(784)
train_dataset = MNIST(
    train=True,
    min_val=min_prob,
    max_val=max_prob,
    offset=noise_offset,
    pixel_permutation=shared_permutation,
)
test_dataset = MNIST(
    train=False,
    min_val=min_prob,
    max_val=max_prob,
    offset=noise_offset,
    pixel_permutation=shared_permutation,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=patch_collate,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=patch_collate,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

# ======================================================================================================================

import tracetorch as tt


class Bar(tt.Model):
    def __init__(self, num_in_out, num_neurons):
        super().__init__()

        self.lin_in = nn.Linear(num_in_out, num_in_out, bias=False)
        nn.init.eye_(self.lin_in.weight)

        coordinates = torch.randn(num_neurons, 5)
        distances = torch.linalg.vector_norm(coordinates, ord=2, dim=-1)
        coordinates = coordinates / distances.unsqueeze(-1)

        out_degrees = torch.empty(num_neurons).log_normal_(2, 1) + 1

        flow_values = torch.ones_like(distances) + (5 / out_degrees)

        self.fdsr = tt.snn.FDSR(
            lif_neurons=tt.snn.LIB(
                num_neurons,
                beta=torch.rand(num_neurons),
                threshold=torch.rand(num_neurons),
                bias=torch.randn(num_neurons) * 0.1,
                quant_fn=nn.Identity(),
            ),
            coordinates=coordinates,
            flow_values=flow_values,
            out_degrees=out_degrees,
            in_features=num_in_out,
            out_features=num_in_out,
            dim=-1,
        )

        self.lin_out = nn.Linear(num_in_out, num_in_out, bias=False)
        nn.init.zeros_(self.lin_out.weight)

    def forward(self, x):
        return x + self.lin_out(self.fdsr(self.lin_in(x)))


class FDSR(tt.Model):
    def __init__(
            self,
            num_neurons=2048,
            num_in_out=128,
            num_layers=2,
    ):
        super().__init__()

        self.enc = nn.Linear(kernel_size ** 2, num_in_out, bias=False)

        self.layers = nn.ModuleList([Bar(
            num_in_out=num_in_out,
            num_neurons=num_neurons,
        ) for _ in range(num_layers)])

        self.dec = nn.Linear(num_in_out, 10, bias=False)

    def forward(self, x):
        x = self.enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dec(x)

        return x


model = FDSR().to(device)


@torch.no_grad()
def update_ema_model(model, ema_model, decay: float = 0.995):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


ema_model = copy.deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
    param.requires_grad = False

print(f"Total: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
loss_fn = nn.functional.cross_entropy

train_losses, train_accs = [], []

num_epochs = 10
for e in range(num_epochs):
    model.train()
    for (img, seq, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        img, seq, label = img.to(device), seq.to(device), label.to(device)

        with torch.no_grad():
            corruption_levels_1d = torch.rand(batch_size).to(device)
            corruption_levels = (corruption_levels_1d ** 0.1).view(1, -1, 1)
            seq = seq * corruption_levels + (torch.randn_like(seq) + 0.5) * (1 - corruption_levels)

        model.zero_grad()
        model.zero_states()

        # Collect outputs from all timesteps
        timestep_outputs = []
        for t in range(seq.size(0)):
            model_output = model(seq[t])
            timestep_outputs.append(model_output)

        # 1. Stack all outputs: [T, B, 10]
        all_outputs = torch.stack(timestep_outputs, dim=0)
        T, B, _ = all_outputs.shape

        # 2. Calculate the raw per-timestep, per-sample loss
        # target needs to be [T, B] so we expand the label
        target = label.unsqueeze(0).expand(T, B)

        # We flatten to [T*B, 10] and [T*B] to use the fast functional cross_entropy
        raw_loss = nn.functional.cross_entropy(
            all_outputs.reshape(-1, 10),
            target.reshape(-1),
            reduction='none'
        ).view(T, B)  # Reshape back to [T, B]

        # 3. Apply Temporal Weighting (Linear ramp)
        # weights: [T, 1]
        weights = (torch.arange(1, T + 1, dtype=torch.float32, device=device).view(T, 1) / T) ** 10

        # Weight the loss and normalize by the mean weight to preserve magnitude
        # (loss * weights).sum() / weights.sum()
        temporal_weighted_loss = (raw_loss * weights).sum(dim=0) / weights.sum()  # Result: [B]

        # 4. Apply Batch Corruption Weighting
        # corruption_levels_1d: [B]
        # We want to trust clean images more than noisy ones.
        batch_weighted_loss = (temporal_weighted_loss * corruption_levels_1d).sum() / corruption_levels_1d.sum()

        # Final scalar loss
        loss = batch_weighted_loss

        # Use final timestep for accuracy (without corruption weighting)
        pred_classes = timestep_outputs[-1].argmax(dim=-1)
        frac_correct = (pred_classes == label).sum().item() / batch_size

        train_losses.append(loss.item())
        train_accs.append(frac_correct)

        loss.backward()
        optimizer.step()
        update_ema_model(model, ema_model, decay=0.995)

    plt.title("LOSS")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.show()

    plt.title("ACC")
    plt.plot(train_accs, label="train")
    plt.legend()
    plt.show()

    model.eval()
    ema_model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (img, seq, label) in tqdm(test_dataloader, total=len(test_dataloader), desc=f"TEST - E{e}"):
            img, seq, label = img.to(device), seq.to(device), label.to(device)

            ema_model.zero_states()

            for t in range(seq.size(0)):
                model_output = ema_model(seq[t])

            loss = loss_fn(model_output, label)
            test_loss += loss.item()
            pred_classes = model_output.argmax(dim=-1)
            frac_correct = (pred_classes == label).sum().item() / batch_size
            test_acc += frac_correct

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        print(f"TEST - Loss: {test_loss} | Acc: {test_acc}")

with torch.no_grad():
    img_batch, seq_batch, label_batch = next(iter(test_dataloader))
    img_batch, seq_batch, label_batch = img_batch.to(device), seq_batch.to(device), label_batch.to(device)

    for i in range(10):
        img, seq, label = img_batch[i], seq_batch[:, i], label_batch[i]  # Extract i-th sample
        ema_model.zero_states()

        input_spike_train = []
        model_outputs = []
        losses = []
        traces = []
        running_loss = 0.0

        for t in range(seq.size(0)):  # Iterate over timesteps
            spk_input = seq[t].unsqueeze(0)  # [1, area]
            input_spike_train.append(spk_input.squeeze(0))  # [area]

            model_output = ema_model(spk_input).squeeze(0)  # [10]
            model_outputs.append(nn.functional.softmax(model_output, dim=-1))
            loss = loss_fn(model_output, label.unsqueeze(0))
            running_loss += loss.item() / seq.size(0)
            losses.append(loss.item())

            trace_thing = ema_model.layers[-1].fdsr.trace.clone().detach()  # Get current synaptic trace
            traces.append(trace_thing.squeeze(0))

        tt.plot.render_image(img.unsqueeze(0), title=f"Digit: {label.item()}")
        # Visualize the input spike train (transpose to [neurons, timesteps] for spike_train)
        input_spike_tensor = torch.stack(input_spike_train).T  # [area, T]
        tt.plot.spike_train([input_spike_tensor.T[t] for t in range(input_spike_tensor.T.size(0))], title="Input")
        tt.plot.spike_train(model_outputs, title="Model Output")
        traces_tensor = torch.stack(traces).T  # [num_neurons, T]
        tt.plot.spike_train([traces_tensor.T[t] for t in range(traces_tensor.T.size(0))], title="Synaptic Trace")
        plt.title(f"Loss over time: {running_loss:.3f} | Final Loss: {losses[-1]:.3f}")
        plt.plot(losses)
        plt.show()
