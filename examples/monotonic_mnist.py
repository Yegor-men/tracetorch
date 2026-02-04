import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================================================================

def one_hot_encode(label):
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


class OneHotMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, min_val=0.0, max_val=1.0, offset=0.0):
        self.dataset = datasets.MNIST(
            root='data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                lambda x: x * (max_val - min_val) + min_val + offset,
            ])
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        one_hot_label = one_hot_encode(label)
        return image, one_hot_label

    def __len__(self):
        return len(self.dataset)


# ======================================================================================================================

NUM_EPOCHS = 1
NUM_TIMESTEPS = 20
BATCH_SIZE = 64
INIT_BETA_DECAY = tt.functional.halflife_to_decay(NUM_TIMESTEPS / 100)


class BaseModel(snn.TTModule):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(1, 16, 5, dilation=2),  # 28-8=20
            snn.LIF(16, INIT_BETA_DECAY, 1.0, dim=-3),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 20/2=10
            nn.Conv2d(16, 32, 3),  # 10-2=8
            snn.LIF(32, INIT_BETA_DECAY, 1.0, dim=-3),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 8/2=4
            nn.Conv2d(32, 64, 4),
            snn.LIF(64, INIT_BETA_DECAY, 1.0, dim=-3),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            snn.LIF(64, INIT_BETA_DECAY, 1.0),
            nn.Linear(64, 10),
            snn.Readout(10, INIT_BETA_DECAY, beta_rank=0),
            nn.Softmax(-1)
        )

    def forward(self, x):
        return self.mlp(x)




class AlternateModel(snn.TTModule):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(1, 16, 5, dilation=2),  # 28-8=20
            # snn.BLIF(16, INIT_BETA_DECAY, 1.0, 1.0, dim=-3),
            snn.RLIF(16, INIT_BETA_DECAY, INIT_BETA_DECAY, 1.0, dim=-3),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 20/2=10
            nn.Conv2d(16, 32, 3),  # 10-2=8
            # snn.BLIF(32, INIT_BETA_DECAY, 1.0, 1.0, dim=-3),
            # snn.BLIF(32, INIT_BETA_DECAY, 1.0, 1.0, dim=-3),
            snn.RLIF(32, INIT_BETA_DECAY, INIT_BETA_DECAY, 1.0, dim=-3),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 8/2=4
            nn.Conv2d(32, 64, 4),
            # snn.BLIF(64, INIT_BETA_DECAY, 1.0, 1.0, dim=-3),
            # snn.BLIF(64, INIT_BETA_DECAY, 1.0, 1.0, dim=-3),
            snn.RLIF(64, INIT_BETA_DECAY, INIT_BETA_DECAY, 1.0, dim=-3),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            # snn.BLIF(64, INIT_BETA_DECAY, 1.0, 1.0),
            # snn.BLIF(64, INIT_BETA_DECAY, 1.0, 1.0),
            snn.RLIF(64, INIT_BETA_DECAY, INIT_BETA_DECAY, 1.0),
            nn.Linear(64, 10),
            snn.Readout(10, INIT_BETA_DECAY, beta_rank=0),
            nn.Softmax(-1)
        )

    def forward(self, x):
        return self.mlp(x)


base_model = BaseModel().to(device)
alternate_model = AlternateModel().to(device)

base_model.zero_states()
alternate_model.zero_states()

ema_base = copy.deepcopy(base_model)
ema_alternate = copy.deepcopy(alternate_model)

ema_decay = 0.999


def update_ema_model(model, ema_model, decay):
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        ema_params = dict(ema_model.named_parameters())

        for name, param in model_params.items():
            ema_param = ema_params[name]
            if ema_param.shape != param.shape:
                raise RuntimeError(
                    f"EMA model parameter shape mismatch for {name}: EMA shape {ema_param.shape}, Model shape {param.shape}")
            ema_param.data.mul_(decay).add_(param.data, alpha=1. - decay)

        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.data.copy_(buffer.data)


base_optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-3)
alternate_optimizer = torch.optim.AdamW(alternate_model.parameters(), lr=1e-3)


def get_model_params(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{name} total parameters: {total_params:,}")


get_model_params(base_model, "BASE")
get_model_params(alternate_model, "alternate")

min_prob, max_prob, noise_offset = 0.0, 1.0, 0.0

train_dataset = OneHotMNIST(train=True, min_val=min_prob, max_val=max_prob, offset=noise_offset)
test_dataset = OneHotMNIST(train=False, min_val=min_prob, max_val=max_prob, offset=noise_offset)

train_dloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======================================================================================================================

loss_fn = tt.loss.soft_cross_entropy
# loss_fn = nn.functional.mse_loss

base_losses = []
alternate_losses = []

for e in range(NUM_EPOCHS):
    base_model.train()
    alternate_model.train()

    for image, label in tqdm(train_dloader, total=len(train_dloader), desc=f"TRAIN - E{e}"):
        image, label = image.to(device), label.to(device)

        base_model.zero_grad()
        alternate_model.zero_grad()

        base_model.zero_states()
        alternate_model.zero_states()

        for timestep in range(NUM_TIMESTEPS):
            base_out = base_model(image)
            alternate_out = alternate_model(image)

        base_loss = loss_fn(base_out, label)
        alternate_loss = loss_fn(alternate_out, label)

        base_losses.append(base_loss.item())
        alternate_losses.append(alternate_loss.item())

        base_loss.backward()
        alternate_loss.backward()

        base_optimizer.step()
        alternate_optimizer.step()

        update_ema_model(base_model, ema_base, ema_decay)
        update_ema_model(alternate_model, ema_alternate, ema_decay)

    plt.title("Train")
    plt.plot(base_losses, label="BASE")
    plt.plot(alternate_losses, label="ALTERNATE")
    plt.legend()
    plt.show()

    ema_base.eval()
    ema_alternate.eval()

    base_model_loss, alternate_model_loss = 0, 0

    with torch.no_grad():
        for image, label in tqdm(test_dloader, total=len(test_dloader), desc=f"TEST - E{e}"):
            image, label = image.to(device), label.to(device)

            ema_base.zero_states()
            ema_alternate.zero_states()

            for timestep in range(NUM_TIMESTEPS):
                base_out = ema_base(image)
                alternate_out = ema_alternate(image)

            base_loss = loss_fn(base_out, label)
            alternate_loss = loss_fn(alternate_out, label)

            base_model_loss += base_loss.item()
            alternate_model_loss += alternate_loss.item()

    base_model_loss /= len(test_dloader)
    alternate_model_loss /= len(test_dloader)

    print(f"BASE: {base_model_loss} | ALTERNATE: {alternate_model_loss}")
