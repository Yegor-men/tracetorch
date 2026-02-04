"""
A test on long sequence manipulations, MNIST

The previous MNIST tutorial focused on if the model could learn to generalize to noisy and clean data using various
training methods, such as full BPTT or BPTT truncated at each timestep, thus getting online learning. We tested both
dense and sparse learning signals, and intentionally lowered the decay values to force the model to learn to accumulate
charge. However, this isn't really an RNN model in the sense. We were just training the model to accumulate charge,
not necessarily do anything useful. The order in which we got the signals didn't really matter, they'd just be
accumulated anyway, and there was no need to manipulate the accumulated signals. If we were looking at this as a
function through time, the previous example is monotonic, while this one is nonmonotonic. If the model can learn to
figure out temporal dynamics from a sparse learning signal, we can hope that the model will be able to learn in RL or
other systems of sparse learning signals. It's still imperfect, in that theoretically the model might somehow learn to
represent each timestep and thus reverse engineer the coordinate, and now it's back to being a monotonic function, but
still, it's a lot better than before and at that point you may raise the argument that the goal of any SNN is to turn
a temporally nonmonotonic function into a monotonic one.

The idea here is that the model inspects the image over 784 timesteps, it's basically sliding a 1x1 kernel across the
image. It effectively gets a long chain of 1s and 0s throughout time that represent the image, and it must learn to map
that long sequence to a classification. Unlike the previous example, here, we don't have a per timestep learning signal,
we don't know at what timestep it becomes obvious what the number is, so the model must somehow store all this in
working memory and figure it out.
"""

# ======================================================================================================================

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
batch_size = 64
kernel_size = 4
stride = 4
pad = False
num_workers = 0
pin_memory = True


def one_hot_encode(label):
    return nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


class OneHotMNISTImage(torch.utils.data.Dataset):
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

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    labels_onehot = nn.functional.one_hot(labels_tensor, num_classes=10).float()  # [B, 10]

    return imgs_b, seq, labels_onehot


train_dataset = OneHotMNISTImage(train=True, min_val=min_prob, max_val=max_prob, offset=noise_offset)
test_dataset = OneHotMNISTImage(train=False, min_val=min_prob, max_val=max_prob, offset=noise_offset)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=patch_collate, num_workers=num_workers, pin_memory=pin_memory)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=patch_collate, num_workers=num_workers, pin_memory=pin_memory)

# ======================================================================================================================

import tracetorch as tt
from tracetorch import snn


class Layer(snn.TTModule):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lif = snn.BSRLIF(
            hidden_dim,
            alpha=torch.rand(hidden_dim),
            beta=torch.rand(hidden_dim),
            gamma=torch.rand(hidden_dim),
            pos_threshold=torch.rand(hidden_dim),
            neg_threshold=torch.rand(hidden_dim),
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.lin(self.lif(x))


class SNN(snn.TTModule):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()

        self.enc = nn.Linear(kernel_size ** 2, hidden_dim)
        self.net = nn.Sequential(*[Layer(hidden_dim) for _ in range(num_layers)])
        self.dec = nn.Sequential(
            snn.SRReadout(
                hidden_dim,
                alpha=torch.rand(hidden_dim),
                beta=torch.rand(hidden_dim),
                gamma=torch.rand(hidden_dim),
            ),
            nn.Linear(hidden_dim, 10),
            nn.Softmax(-1)
        )
        nn.init.zeros_(self.dec[-2].weight)
        nn.init.zeros_(self.dec[-2].bias)

    def forward(self, x):
        return self.dec(self.net(self.enc(x)))


model = SNN(128, 3).to(device)
print(f"\nNum params: {model.get_param_count():,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-3)

loss_fn = tt.loss.soft_cross_entropy
# loss_fn = nn.functional.mse_loss

train_losses, train_accs = [], []

num_epochs = 10
for e in range(num_epochs):
    model.train()
    for (img, seq, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        img, seq, label = img.to(device), torch.bernoulli(seq).to(device), label.to(device)

        model.zero_grad()
        model.zero_states()

        for t in range(seq.size(0)):
            model_output = model(seq[t])

        loss = loss_fn(model_output, label)
        pred_classes = model_output.argmax(dim=-1)
        true_classes = label.argmax(dim=-1)
        frac_correct = (pred_classes == true_classes).sum().item() / batch_size

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
            true_classes = label.argmax(dim=-1)
            frac_correct = (pred_classes == true_classes).sum().item() / batch_size
            test_acc += frac_correct

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        print(f"TEST - Loss: {test_loss} | Acc: {test_acc}")
