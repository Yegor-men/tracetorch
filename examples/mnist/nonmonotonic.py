import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# ======================================================================================================================

min_prob, max_prob, noise_offset = 0.0, 1.0, 0.0
batch_size = 100
kernel_size = 4
stride = 4
pad = False
num_workers = 0
pin_memory = True
shuffle_pixels = True  # Toggle this for sequence complexity


class MNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, min_val=0.0, max_val=1.0, offset=0.0, root="data",
                 pixel_permutation=None):
        if pixel_permutation is None:
            self.pixel_permutation = torch.randperm(784)
        else:
            self.pixel_permutation = pixel_permutation

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
        img, label = self.ds[idx]

        if shuffle_pixels:
            C, H, W = img.shape
            img_flat = img.view(-1)
            img_shuffled = img_flat[self.pixel_permutation]
            img = img_shuffled.view(C, H, W)

        return img, label


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


shared_permutation = torch.randperm(784)
train_dataset = MNIST(train=True, pixel_permutation=shared_permutation)
test_dataset = MNIST(train=False, pixel_permutation=shared_permutation)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=patch_collate,
                              num_workers=num_workers, pin_memory=pin_memory)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=patch_collate,
                             num_workers=num_workers, pin_memory=pin_memory)

# ======================================================================================================================

import tracetorch as tt


class SSM(tt.Model):
    def __init__(self, working_dim, num_layers):
        super().__init__()
        self.enc = nn.Linear(kernel_size ** 2, working_dim)

        self.layers = nn.ModuleList([
            tt.ssm.S6(
                num_neurons=working_dim,
                d_state=16,
            ) for _ in range(num_layers)
        ])

        self.dec = nn.Linear(working_dim, 10)
        nn.init.zeros_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        x = self.enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dec(x)
        return x


model = SSM(working_dim=128, num_layers=4).to(device)
print(f"Total: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
loss_fn = nn.functional.cross_entropy

train_losses, train_accs = [], []
num_epochs = 5

for e in range(num_epochs):
    model.train()
    for (img, seq, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        img, seq, label = img.to(device), seq.to(device), label.to(device)

        # Removed dynamic noise to focus on architecture validation
        model.zero_grad()
        model.zero_states()

        timestep_outputs = []
        for t in range(seq.size(0)):
            model_output = model(seq[t])
            timestep_outputs.append(model_output)

        final_outputs = timestep_outputs[-1]
        loss = loss_fn(final_outputs, label)

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
        img, seq, label = img_batch[i], seq_batch[:, i], label_batch[i]
        model.zero_states()

        input_spike_train = []
        model_outputs = []
        losses = []
        running_loss = 0.0

        for t in range(seq.size(0)):
            spk_input = seq[t].unsqueeze(0)
            input_spike_train.append(spk_input.squeeze(0))

            model_output = model(spk_input).squeeze(0)
            model_outputs.append(nn.functional.softmax(model_output, dim=-1))
            loss = loss_fn(model_output, label.unsqueeze(0))
            running_loss += loss.item() / seq.size(0)
            losses.append(loss.item())

        tt.plot.render_image(img.unsqueeze(0), title=f"Digit: {label.item()}")
        input_spike_tensor = torch.stack(input_spike_train).T
        tt.plot.spike_train([input_spike_tensor.T[t] for t in range(input_spike_tensor.T.size(0))], title="Input")
        tt.plot.spike_train(model_outputs, title="Model Output")
        plt.title(f"Loss over time | Loss: {running_loss:.3f}")
        plt.plot(losses)
        plt.show()
