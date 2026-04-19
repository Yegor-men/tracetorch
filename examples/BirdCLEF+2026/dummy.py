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

min_scale = tt.functional.halflife_to_decay(16 / (kernel_size ** 2))  # for 4x4 kernel, decay must be over 1 step
max_scale = tt.functional.halflife_to_decay(784 / (kernel_size ** 2))  # for 1x1 kernel, halflife must be all steps
scale_diff = max_scale - min_scale

weight_scale = (kernel_size ** 2) / 16  # must be 1 when 4x4, and 1/16 when 1x1 since it's 16x less per step


class Foo(tt.Model):
    def __init__(self, working_dim):
        super().__init__()

        self.lin_in = nn.Linear(working_dim, working_dim, bias=False)
        nn.init.eye_(self.lin_in.weight)

        self.lif = tt.snn.LIB(
            num_neurons=working_dim,
            beta=torch.rand(working_dim),
            threshold=torch.rand(working_dim),
            bias=torch.randn(working_dim) * 0.1,
            quant_fn=nn.Identity(),
        )

        self.lin_out = nn.Linear(working_dim, working_dim, bias=False)
        nn.init.zeros_(self.lin_out.weight)

    def forward(self, x):
        return x + self.lin_out(self.lif(self.lin_in(x)))


class SSM(tt.Model):
    def __init__(self, working_dim, d_state, num_layers):
        super().__init__()

        self.enc = nn.Linear(kernel_size ** 2, working_dim)

        # self.layers = nn.ModuleList([
        #     tt.ssm.SelectiveSSM(
        #         num_neurons=working_dim,
        #         d_state=d_state,
        #     ) for _ in range(num_layers)
        # ])

        self.layers = nn.ModuleList([Foo(working_dim) for _ in range(num_layers)])

        self.dec = nn.Linear(working_dim, 10)
        nn.init.zeros_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        x = self.enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dec(x)
        return x


class Bar(tt.Model):
    def __init__(self, working_dim, num_neurons, num_connections, num_dims, flow):
        super().__init__()

        self.lin_in = nn.Linear(working_dim, working_dim, bias=False)
        nn.init.eye_(self.lin_in.weight)

        self.fdsr = tt.snn.FDSR(
            in_features=working_dim,
            out_features=working_dim,
            num_neurons=num_neurons,
            num_connections=num_connections,
            gamma=torch.rand(num_neurons) * scale_diff + min_scale,
            num_dims=num_dims,
            flow=flow,
            dim=-1,
        )

        self.lin_out = nn.Linear(working_dim, working_dim, bias=False)
        nn.init.zeros_(self.lin_out.weight)

    def forward(self, x):
        return x + self.lin_out(self.fdsr(self.lin_in(x)))


class FDSR(tt.Model):
    def __init__(
            self,
            working_dim: int,
            num_neurons: int,
            num_connections: int,
            num_dims: int,
            flow: float,
            num_layers: int,
    ):
        super().__init__()

        self.enc = nn.Linear(kernel_size ** 2, working_dim, bias=False)

        self.layers = nn.ModuleList([Bar(
            working_dim=working_dim,
            num_neurons=num_neurons,
            num_connections=num_connections,
            num_dims=num_dims,
            flow=flow,
        ) for _ in range(num_layers)])

        self.dec = nn.Linear(working_dim, 10, bias=False)

    def forward(self, x):
        x = self.enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dec(x)

        return x


model = FDSR(working_dim=128, num_neurons=512, num_connections=64, num_dims=4, flow=0.1, num_layers=5).to(device)
# model = SSM(working_dim=128, d_state=16, num_layers=10).to(device)

print(f"Total: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
loss_fn = nn.functional.cross_entropy

train_losses, train_accs = [], []

num_epochs = 10
for e in range(num_epochs):
    model.train()
    for (img, seq, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        img, seq, label = img.to(device), seq.to(device), label.to(device)

        # Apply dynamic corruption per image in batch
        batch_size = img.size(0)
        corruption_levels = torch.rand(batch_size, device=device)  # [0,1] values
        sqrt_corruption = torch.sqrt(corruption_levels)  # sqrt values for image weighting

        # Generate noise for each image
        noise = torch.rand_like(img)  # Same shape as images

        # Apply corruption: image*sqrt(x) + noise*(1-sqrt(x))
        # Need to expand sqrt_corruption to match image dimensions
        sqrt_correlation = sqrt_corruption.view(batch_size, 1, 1, 1)  # [B,1,1,1]
        img_corrupted = img * sqrt_correlation + noise * (1 - sqrt_correlation)

        # Update seq with corrupted images
        # Need to re-run patch_collate logic here with corrupted images
        B, C, H, W = img_corrupted.shape
        patches = img_corrupted.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        n_h = patches.size(1)
        n_w = patches.size(2)
        n_patches = n_h * n_w
        patches = patches.view(B, n_patches, C * kernel_size * kernel_size)
        seq_corrupted = patches.permute(1, 0, 2).contiguous()  # [T, B, area]

        model.zero_grad()
        model.zero_states()

        # Collect outputs from all timesteps
        timestep_outputs = []
        for t in range(seq_corrupted.size(0)):
            model_output = model(seq_corrupted[t])
            timestep_outputs.append(model_output)

        # Calculate weighted loss for each timestep
        T = len(timestep_outputs)
        weights = torch.arange(1, T + 1, dtype=torch.float32, device=device) / T

        # Calculate per-image losses with corruption scaling
        final_outputs = timestep_outputs[-1]  # [B, 10]
        per_image_losses = torch.stack([
            loss_fn(final_outputs[i:i + 1], label[i:i + 1]) * corruption_levels[i]
            for i in range(batch_size)
        ])

        loss = per_image_losses.mean()

        # Use final timestep for accuracy (without corruption weighting)
        pred_classes = final_outputs.argmax(dim=-1)
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
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (img, seq, label) in tqdm(test_dataloader, total=len(test_dataloader), desc=f"TEST - E{e}"):
            img, seq, label = img.to(device), seq.to(device), label.to(device)

            model.zero_states()

            for t in range(seq.size(0)):
                model_output = model(seq[t])

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
        model.zero_states()

        input_spike_train = []
        model_outputs = []
        losses = []
        traces = []
        running_loss = 0.0

        for t in range(seq.size(0)):  # Iterate over timesteps
            spk_input = seq[t].unsqueeze(0)  # [1, area]
            input_spike_train.append(spk_input.squeeze(0))  # [area]

            model_output = model(spk_input).squeeze(0)  # [10]
            model_outputs.append(nn.functional.softmax(model_output, dim=-1))
            loss = loss_fn(model_output, label.unsqueeze(0))
            running_loss += loss.item() / seq.size(0)
            losses.append(loss.item())

            trace_thing = model.layers[-1].fdsr.trace.clone().detach()  # Get current synaptic trace
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
