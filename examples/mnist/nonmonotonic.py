import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

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
                transforms.ToTensor(),  # -> [C, H, W] in [0,1]
                transforms.Lambda(lambda x: x * (max_val - min_val) + min_val + offset)
            ])
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]  # img: [C, H, W], label: int
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


train_dataset = MNIST(train=True, min_val=min_prob, max_val=max_prob, offset=noise_offset)
test_dataset = MNIST(train=False, min_val=min_prob, max_val=max_prob, offset=noise_offset)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=patch_collate, num_workers=num_workers, pin_memory=pin_memory)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=patch_collate, num_workers=num_workers, pin_memory=pin_memory)

# ======================================================================================================================

import tracetorch as tt
from tracetorch import snn

min_scale = tt.functional.halflife_to_decay(16 / (kernel_size ** 2))  # for 4x4 kernel, decay must be over 1 step
max_scale = tt.functional.halflife_to_decay(784 / (kernel_size ** 2))  # for 1x1 kernel, halflife must be all steps
scale_diff = max_scale - min_scale

weight_scale = (kernel_size ** 2) / 16  # must be 1 when 4x4, and 1/16 when 1x1 since it's 16x less per step


class ResidualLayer(snn.TTModel):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lif = snn.DSRLITS(
            hidden_dim,
            pos_alpha=torch.rand(hidden_dim) * scale_diff + min_scale,
            neg_alpha=torch.rand(hidden_dim) * scale_diff + min_scale,
            pos_beta=torch.rand(hidden_dim) * scale_diff + min_scale,
            neg_beta=torch.rand(hidden_dim) * scale_diff + min_scale,
            pos_gamma=torch.rand(hidden_dim) * scale_diff + min_scale,
            neg_gamma=torch.rand(hidden_dim) * scale_diff + min_scale,
            pos_threshold=torch.rand(hidden_dim),
            neg_threshold=torch.rand(hidden_dim),
            pos_scale=torch.rand(hidden_dim),
            neg_scale=torch.rand(hidden_dim),
            pos_rec_weight=torch.randn(hidden_dim) * 0.1 * weight_scale,
            neg_rec_weight=torch.randn(hidden_dim) * 0.1 * weight_scale,
            quant_fn="probabilistic",
        )
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.Tanh()

    def forward(self, x):
        return self.norm(x + self.lin2(self.lif(self.lin1(x))))


class DecoderTransformer(nn.Module):
    def __init__(self, emb_dim: int = 128, num_tokens: int = 11):
        super().__init__()
        self.num_tokens = num_tokens
        self.mha = nn.MultiheadAttention(emb_dim, 4, 0.0, batch_first=True)
        self.mha_scalar = nn.Parameter(torch.ones(emb_dim) * 0.1)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        seq_len = self.num_tokens
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

        attn_out, _ = self.mha(x, x, x, attn_mask=causal_mask)
        x = self.norm(x + attn_out * self.mha_scalar)
        return x


class SNN(snn.TTModel):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 10, num_decoder_blocks: int = 2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tokens = num_layers + 1  # 10 layers + initial state = 11 tokens

        self.enc = nn.Sequential(nn.Linear(kernel_size ** 2, hidden_dim), nn.Tanh())
        self.layers = nn.ModuleList([
            ResidualLayer(hidden_dim=hidden_dim) for _ in range(num_layers)
        ])

        self.token_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_tokens)
        ])

        self.token_norm = nn.LayerNorm(hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            DecoderTransformer(emb_dim=hidden_dim, num_tokens=self.num_tokens)
            for _ in range(num_decoder_blocks)
        ])

        self.dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            snn.DSLI(
                hidden_dim,
                pos_beta=torch.rand(hidden_dim),
                neg_beta=torch.rand(hidden_dim),
                pos_alpha=torch.rand(hidden_dim),
                neg_alpha=torch.rand(hidden_dim),
            ),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 10)
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        x = self.enc(x)

        # Accumulate intermediate states: [B, 11, H]
        intermediate_states = []

        # Store initial state before any blocks (token 0)
        intermediate_states.append(self.token_proj[-1](x.clone()))

        # Process through blocks and store intermediate states
        for i, block in enumerate(self.layers):
            x = block(x)
            intermediate_states.append(self.token_proj[i](x.clone()))

        # Stack all intermediate states: [B, 11, H]
        token_sequence = torch.stack(intermediate_states, dim=1)  # [B, 11, H]
        token_sequence = self.token_norm(token_sequence)
        # Pass through decoder transformer blocks
        for decoder_block in self.decoder_blocks:
            token_sequence = decoder_block(token_sequence)  # [B, 11, H]

        # Use the final token (most processed) for classification
        final_token = token_sequence[:, -1, :]  # [B, H]
        output = self.dec(final_token)  # [B, 10]

        return output


class DynamicLayer(snn.TTLayer):
    def __init__(self, hidden_features: int, d_state: int, dim: int = -1):
        super().__init__(d_state, dim)

        self.A = nn.Linear(hidden_features, d_state)
        self.scales = nn.Parameter(torch.randn(d_state))
        self.B = nn.Linear(hidden_features, d_state)
        self.C = nn.Linear(d_state, hidden_features)
        self.D = nn.Linear(hidden_features, hidden_features)
        nn.init.zeros_(self.D.bias)

        self._initialize_state("mem")

    def forward(self, x):
        x = self._to_working_dim(x)

        decay = torch.exp(nn.functional.softplus(self.A(x)) * -torch.exp(self.scales))
        self._ensure_states(decay)

        self.mem = self.mem * decay + self.B(x) * (1 - decay)

        out = x + nn.functional.silu(self.D(x)) * self.C(self.mem)

        out = self._from_working_dim(out)

        return out


class Test(snn.TTModel):
    def __init__(self, hidden_dim, d_state, num_layers):
        super().__init__()

        self.enc = nn.Linear(kernel_size ** 2, hidden_dim)
        self.layers = nn.ModuleList([DynamicLayer(hidden_dim, d_state) for _ in range(num_layers)])
        self.dec = nn.Linear(hidden_dim, 10)
        nn.init.zeros_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        x = self.enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dec(x)
        return x


model = Test(hidden_dim=128, d_state=128, num_layers=10).to(device)
# model = SNN(hidden_dim=128, num_layers=10, num_decoder_blocks=2).to(device)
total_params = sum(p.numel() for p in model.parameters())
snn_params = model.get_param_count()
print(f"Total: {total_params:,} -> SNN: {snn_params:,} | Non-SNN: {total_params - snn_params:,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

loss_fn = nn.functional.cross_entropy

train_losses, train_accs = [], []

num_epochs = 5
for e in range(num_epochs):
    model.train()
    for (img, seq, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        img, seq, label = img.to(device), torch.bernoulli(seq).to(device), label.to(device)

        model.zero_grad()
        model.zero_states()

        # Collect outputs from all timesteps
        timestep_outputs = []
        for t in range(seq.size(0)):
            model_output = model(seq[t])
            timestep_outputs.append(model_output)

        # Calculate loss for each timestep
        T = len(timestep_outputs)
        weights = torch.arange(1, T + 1, dtype=torch.float32, device=device) / T

        losses = torch.stack([loss_fn(output, label) for output in timestep_outputs])
        loss = (losses * weights).sum() / weights.sum()  # Divide by sum of weights

        # Use final timestep for accuracy
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
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (img, seq, label) in tqdm(test_dataloader, total=len(test_dataloader), desc=f"TEST - E{e}"):
            img, seq, label = img.to(device), torch.bernoulli(seq).to(device), label.to(device)

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
        running_loss = 0.0

        for t in range(seq.size(0)):  # Iterate over timesteps
            spk_input = torch.bernoulli(seq[t]).unsqueeze(0)  # [1, area]
            input_spike_train.append(spk_input.squeeze(0))  # [area]

            model_output = model(spk_input).squeeze(0)  # [10]
            model_outputs.append(nn.functional.softmax(model_output, dim=-1))
            loss = loss_fn(model_output, label.unsqueeze(0))
            running_loss += loss.item() / seq.size(0)
            losses.append(loss.item())

        tt.plot.render_image(img.unsqueeze(0), title=f"Loss: {running_loss:.3f}")
        # Visualize the input spike train (transpose to [neurons, timesteps] for spike_train)
        input_spike_tensor = torch.stack(input_spike_train).T  # [area, T]
        tt.plot.spike_train([input_spike_tensor.T[t] for t in range(input_spike_tensor.T.size(0))], title="Input")
        tt.plot.spike_train(model_outputs, title="Model Output")
        plt.title("Loss over time")
        plt.plot(losses)
        plt.show()
