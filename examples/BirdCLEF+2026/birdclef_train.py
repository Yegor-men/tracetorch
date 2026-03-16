import os
import copy
import random
import pandas as pd
import torchaudio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from safetensors.torch import save_file

import tracetorch as tt
from tracetorch import snn

# ---------------------- basic config ----------------------
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.manual_seed_all(0)


# =====================================================================
# 1. Dataset & Dataloader
# =====================================================================
class BirdCLEFDataset(Dataset):
    def __init__(self, data_dir='data/birdclef-2026', split='train', duration=5.0, sr=32000):
        self.data_dir = data_dir
        self.sr = sr
        self.num_samples = int(duration * sr)  # 5 sec * 32000 = 160,000 samples

        self.audio_dir = os.path.join(data_dir, "train_audio")
        self.csv_path = os.path.join(data_dir, "train.csv")

        if not os.path.exists(self.csv_path) or not os.path.exists(self.audio_dir):
            raise FileNotFoundError(
                f"\n[ERROR] Could not find the dataset at {data_dir}.\n"
                f"Please ensure:\n"
                f"1. {self.csv_path} exists.\n"
                f"2. {self.audio_dir}/ exists."
            )

        # EXACTLY 1024 frequency bins and 313 time steps
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=2046,  # 2046 // 2 + 1 = 1024 bins
            hop_length=512,  # 160000 / 512 = 312.5 -> 313 frames
            power=2.0
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        self.samples = []
        df_full = pd.read_csv(self.csv_path)

        # Auto-detect column names since Kaggle changes them slightly
        filename_col = 'filename' if 'filename' in df_full.columns else df_full.columns[0]
        label_col = 'primary_label' if 'primary_label' in df_full.columns else df_full.columns[1]

        # Sort classes alphabetically to ensure indices are perfectly stable across runs
        self.classes = sorted(df_full[label_col].astype(str).unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 80/20 Train/Test split
        if split == 'train':
            df = df_full.sample(frac=0.8, random_state=42)
        else:
            df = df_full.drop(df_full.sample(frac=0.8, random_state=42).index)

        for _, row in df.iterrows():
            file_path = os.path.join(self.audio_dir, str(row[filename_col]))
            # Some CSVs don't include the .ogg extension, add it if missing
            if not file_path.endswith('.ogg'):
                file_path += '.ogg'

            self.samples.append((
                file_path,
                self.class_to_idx[str(row[label_col])]
            ))

        print(f"Loaded {split} split: {len(self.samples)} real audio samples spanning {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label_idx = self.samples[idx]

        try:
            waveform, sr = torchaudio.load(file_path)
            # Standardize to correct sample rate
            if sr != self.sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.sr)
            # Mix down to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        except Exception:
            # Failsafe for missing or corrupted audio files in the dataset
            waveform = torch.zeros(1, self.num_samples)

        # Random crop or pad to exactly 5 seconds
        if waveform.shape[1] > self.num_samples:
            start = random.randint(0, waveform.shape[1] - self.num_samples)
            waveform = waveform[:, start:start + self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            pad = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # STFT ->[1024, 313]
        spec = self.spec_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0)

        # Standardize frequencies (crucial for SNN stability)
        spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-6)

        X = spec_db.transpose(0, 1)  # [L, E] -> [313, 1024]
        Y = torch.tensor(label_idx, dtype=torch.long)
        return X, Y


def collate_fn(batch):
    X, Y = zip(*batch)
    X = torch.stack(X).transpose(0, 1)  # -> [L, B, 1024]
    Y = torch.stack(Y)  # -> [B]
    return X, Y


# =====================================================================
# 2. traceTorch Architecture
# =====================================================================
class Foobar(nn.Module):
    def __init__(self, slope: float = 4.0):
        super().__init__()
        self.slope = nn.Parameter(torch.log(torch.expm1(torch.tensor([slope]))))

    def forward(self, x):
        return nn.functional.sigmoid(nn.functional.softplus(self.slope) * x)


def foobar(x):
    return nn.functional.sigmoid(2.0 * x)


class ResidualSpike(snn.TTModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lif = snn.DSRLITS(
            hidden_dim,
            pos_alpha=torch.rand(hidden_dim),
            neg_alpha=torch.rand(hidden_dim),
            pos_beta=torch.rand(hidden_dim),
            neg_beta=torch.rand(hidden_dim),
            pos_gamma=torch.rand(hidden_dim),
            neg_gamma=torch.rand(hidden_dim),
            pos_threshold=torch.rand(hidden_dim),
            neg_threshold=torch.rand(hidden_dim),
            pos_scale=torch.randn(hidden_dim) * 0.5 + 1.0,
            neg_scale=torch.randn(hidden_dim) * 0.5 + 1.0,
            pos_rec_weight=torch.randn(hidden_dim) * 0.1,
            neg_rec_weight=torch.randn(hidden_dim) * 0.1,
            spike_fn=foobar,
            deterministic=True,
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        spk = self.lif(x)
        delta = self.lin(spk)
        return x + delta


class SNN(snn.TTModel):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            num_layers: int,
            out_features: int,
    ):
        super().__init__()

        # Replaced Embedding with a Linear projection since we are feeding continuous frequencies
        self.enc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Dropout(0.0),
        )

        layers = [ResidualSpike(hidden_features) for _ in range(num_layers)]
        self.net = nn.Sequential(*layers)

        self.dec = nn.Sequential(
            snn.DSLI(
                hidden_features,
                pos_alpha=torch.rand(hidden_features),
                neg_alpha=torch.rand(hidden_features),
                pos_beta=torch.rand(hidden_features),
                neg_beta=torch.rand(hidden_features),
            ),
            nn.Dropout(0.0),
            nn.Linear(hidden_features, out_features)
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        return self.dec(self.net(self.enc(x)))


def add_spec_noise(tensor: torch.Tensor, p: float = 0.1, device=None) -> torch.Tensor:
    if p <= 0: return tensor
    if device is None: device = tensor.device
    mask = torch.rand(tensor.shape, device=device) > p
    return tensor * mask


@torch.no_grad()
def update_ema_model(model, ema_model, decay: float = 0.999):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


# =====================================================================
# 3. Training Loop
# =====================================================================
if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'tensorboard/birdclef_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)

    batch_size = 32  # Batch size 32 easily handles[1024, 313] sequences on consumer hardware
    minibatch_size = 1

    train_ds = BirdCLEFDataset(split='train')
    val_ds = BirdCLEFDataset(split='val')
    num_classes = len(train_ds.classes)

    # Note: num_workers set to 4 to speed up audio loading
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = SNN(in_features=1024, hidden_features=1024, num_layers=10, out_features=num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    snn_params = model.get_param_count()
    print(f"Total: {total_params:,} -> SNN: {snn_params:,} | Non-SNN: {total_params - snn_params:,}")

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), 1e-5)
    loss_fn = nn.CrossEntropyLoss()

    optimizer_steps = 0
    accum_steps = 0
    num_epochs = 10

    total_steps = num_epochs * max(1, len(train_dataloader)) // minibatch_size
    warmup_steps = max(1, total_steps // 10)
    cosine_steps = total_steps - warmup_steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    for e in range(num_epochs):
        model.train()
        ema_model.eval()

        train_loss_accum = 0.0
        train_correct_accum = 0
        train_samples_accum = 0

        for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"TRAIN - E{e}"):
            x = x.to(device)  # [313, B, 1024]
            y = y.to(device)  # [B]

            # Frequency dropout as data augmentation
            x = add_spec_noise(x, p=0.1)

            model.detach_states()
            model.zero_states()

            # Forward pass over every timestep sequentially
            for t in range(x.size(0)):
                x_t = x[t]
                model_output = model(x_t)

            # ONLY compute loss and accuracy at the final timestep
            loss = loss_fn(model_output, y)

            scaled_loss = loss / minibatch_size
            scaled_loss.backward()

            accum_steps += 1
            train_loss_accum += loss.item()

            preds = model_output.argmax(dim=-1)
            train_correct_accum += (preds == y).sum().item()
            train_samples_accum += y.size(0)

            if accum_steps % minibatch_size == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("learning_rate", current_lr, optimizer_steps)

                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                update_ema_model(model, ema_model)
                optimizer.zero_grad()

                # Log Train Loss and Accuracy
                avg_train_loss = train_loss_accum / minibatch_size
                train_acc = train_correct_accum / train_samples_accum

                writer.add_scalars("loss", {"train": avg_train_loss}, optimizer_steps)
                writer.add_scalars("accuracy", {"train": train_acc}, optimizer_steps)

                # Reset accumulators
                train_loss_accum = 0.0
                train_correct_accum = 0
                train_samples_accum = 0

            # Validation Loop every 400 steps
            if optimizer_steps % 400 == 0 and optimizer_steps != 0:
                val_loss = 0.0
                correct = 0
                total_samples = 0

                ema_model.eval()
                with torch.no_grad():
                    for j, (x, y) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="VAL"):
                        x = x.to(device)
                        y = y.to(device)
                        ema_model.zero_states()

                        for t in range(x.size(0)):
                            model_output = ema_model(x[t])

                        # ONLY compute loss and accuracy at the final timestep
                        loss = loss_fn(model_output, y)
                        val_loss += loss.item()

                        preds = model_output.argmax(dim=-1)
                        correct += (preds == y).sum().item()
                        total_samples += y.size(0)

                if len(val_dataloader) > 0:
                    val_loss /= len(val_dataloader)
                    accuracy = correct / total_samples

                    writer.add_scalars("loss", {"val": val_loss}, optimizer_steps)
                    writer.add_scalars("accuracy", {"val": accuracy}, optimizer_steps)

                    os.makedirs("checkpoints", exist_ok=True)
                    model_path = os.path.join("checkpoints", f"birdclef_step_{optimizer_steps}_ema.safetensors")
                    save_file(ema_model.state_dict(), model_path)
                    print(f"\nSaved model checkpoint to {model_path} | Val Acc: {accuracy:.4f}\n")

    writer.close()
