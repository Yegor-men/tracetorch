import os
import copy
import random
import pandas as pd
import numpy as np
import torchaudio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from safetensors.torch import save_file
import librosa
from sklearn.metrics import roc_auc_score

import tracetorch as tt
from tracetorch import snn

# ---------------------- basic config ----------------------
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.manual_seed_all(0)


# =====================================================================
# Evaluation Metric Helper
# =====================================================================
def compute_macro_roc_auc(preds, targets):
    binary_targets = (targets > 0.0).astype(np.float32)
    target_sums = binary_targets.sum(axis=0)

    scored_columns = np.where((target_sums > 0) & (target_sums < len(targets)))[0]
    if len(scored_columns) == 0:
        return 0.5

    return roc_auc_score(binary_targets[:, scored_columns], preds[:, scored_columns], average='macro')


# =====================================================================
# 1. Dataset & Dataloader (Full 60s Soundscapes)
# =====================================================================
class FullSoundscapeDataset(Dataset):
    def __init__(self, df_labels, class_to_idx, data_dir='data/birdclef-2026', sr=32000):
        self.sr = sr
        # Exactly 60 seconds
        self.num_samples = int(60.0 * sr)
        self.audio_dir = os.path.join(data_dir, "train_soundscapes")
        self.class_to_idx = class_to_idx

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=2048, hop_length=512, n_mels=256, power=2.0)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # Group labels by filename
        self.samples = []
        unique_files = df_labels['filename'].unique()

        for filename in unique_files:
            file_path = os.path.join(self.audio_dir, str(filename))
            if not file_path.endswith('.ogg'): file_path += '.ogg'

            # Target matrix: 12 chunks (5s each), num_classes
            Y_matrix = torch.zeros((12, len(self.class_to_idx)), dtype=torch.float32)

            file_df = df_labels[df_labels['filename'] == filename]
            for _, row in file_df.iterrows():
                # start column is 0, 5, 10, etc.
                time_parts = str(row['start']).split(':')
                start_sec = sum(float(x) * (60 ** i) for i, x in enumerate(reversed(time_parts)))
                # start_sec = float(row['start'])
                chunk_idx = int(start_sec // 5)

                if chunk_idx >= 12: continue

                birds_str = str(row['primary_label'])
                birds = birds_str.split() if ' ' in birds_str else [birds_str]
                for b in birds:
                    if b in self.class_to_idx:
                        Y_matrix[chunk_idx, self.class_to_idx[b]] = 1.0

            self.samples.append((file_path, Y_matrix))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, Y_matrix = self.samples[idx]

        try:
            # Load exact 60 seconds
            waveform_np, _ = librosa.load(file_path, sr=self.sr, mono=True, duration=60.0)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        except Exception:
            waveform = torch.zeros(1, self.num_samples)

        if waveform.shape[1] < self.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))

        spec = self.mel_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0)
        X = ((spec_db + 80.0) / 80.0).transpose(0, 1)  # [~3750, 256]

        return X, Y_matrix


def collate_fn(batch):
    X, Y = zip(*batch)
    X = torch.stack(X).transpose(0, 1)  # [3750, B, 256]
    Y = torch.stack(Y)  # [B, 12, num_classes]
    return X, Y


# =====================================================================
# 2. traceTorch Architecture (Unchanged)
# =====================================================================
class Foobar(nn.Module):
    def __init__(self, slope: float = 4.0):
        super().__init__()
        self.slope = nn.Parameter(torch.log(torch.expm1(torch.tensor([slope]))))

    def forward(self, x):
        return nn.functional.sigmoid(nn.functional.softplus(self.slope) * x)


def foobar(x):
    return nn.functional.sigmoid(4.0 * x)


dsrlits_min_timescale = tt.functional.halflife_to_decay(1)
dsrlits_max_timescale = tt.functional.halflife_to_decay(50)
dsrlits_diff = dsrlits_max_timescale - dsrlits_min_timescale
dsli_timescale = tt.functional.halflife_to_decay(150)


class ResidualSpike(snn.TTModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lif = snn.DSRLITS(
            hidden_dim,
            pos_alpha=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            neg_alpha=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            pos_beta=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            neg_beta=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            pos_gamma=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            neg_gamma=torch.rand(hidden_dim) * dsrlits_diff + dsrlits_min_timescale,
            pos_threshold=torch.rand(hidden_dim),
            neg_threshold=torch.rand(hidden_dim),
            pos_scale=torch.randn(hidden_dim) * 0.5 + 1.0,
            neg_scale=torch.randn(hidden_dim) * 0.5 + 1.0,
            pos_rec_weight=torch.randn(hidden_dim) * 0.1,
            neg_rec_weight=torch.randn(hidden_dim) * 0.1,
            spike_fn=foobar,
            deterministic=False,
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return x + self.lin(self.lif(x))


class SNN(snn.TTModel):
    def __init__(self, in_features, hidden_features, num_layers, out_features):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_features, hidden_features), nn.Dropout(0.0))
        self.net = nn.Sequential(*[ResidualSpike(hidden_features) for _ in range(num_layers)])
        self.dec = nn.Sequential(
            snn.DSLI(
                hidden_features,
                pos_alpha=torch.rand(hidden_features) * dsrlits_max_timescale,
                neg_alpha=torch.rand(hidden_features) * dsrlits_max_timescale,
                pos_beta=torch.rand(hidden_features) * (1 - dsli_timescale) + dsli_timescale,
                neg_beta=torch.rand(hidden_features) * (1 - dsli_timescale) + dsli_timescale,
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
# 3. Training & Validation Loop
# =====================================================================
if __name__ == '__main__':
    import subprocess, webbrowser, time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'tensorboard/birdclef_60s_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)

    # --- Setup Datasets ---
    data_dir = 'data/birdclef-2026'

    # Still load focal train.csv JUST to get the absolute list of 234 classes
    df_focal_all = pd.read_csv(os.path.join(data_dir, "train.csv"))
    label_col = 'primary_label' if 'primary_label' in df_focal_all.columns else df_focal_all.columns[1]
    classes = sorted(df_focal_all[label_col].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    # Load Soundscapes Labels
    ss_csv_path = os.path.join(data_dir, "train_soundscapes_labels.csv")
    df_ss_all = pd.read_csv(ss_csv_path)

    # Get unique labeled files (approx 123)
    unique_ss_files = df_ss_all['filename'].unique()
    np.random.shuffle(unique_ss_files)

    # Split the ~123 files into 80/20 train/val
    split_idx = int(len(unique_ss_files) * 0.8)
    train_files = unique_ss_files[:split_idx]
    val_files = unique_ss_files[split_idx:]

    df_ss_train = df_ss_all[df_ss_all['filename'].isin(train_files)]
    df_ss_val = df_ss_all[df_ss_all['filename'].isin(val_files)]

    train_ds = FullSoundscapeDataset(df_ss_train, class_to_idx, data_dir=data_dir)
    val_ds = FullSoundscapeDataset(df_ss_val, class_to_idx, data_dir=data_dir)

    print(f"Training on {len(train_ds)} 60s soundscapes.")
    print(f"Validating on {len(val_ds)} 60s soundscapes.")

    # Extremely low batch size to prevent 3750-timestep OOM
    # Accumulate 15 times to get an effective batch size of 60
    batch_size = 4
    grad_accum_steps = 15

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = SNN(in_features=256, hidden_features=1024, num_layers=10, out_features=num_classes).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters(): param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), 1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer_steps, accum_steps, num_epochs = 0, 0, 10
    total_steps = num_epochs * max(1, len(train_dataloader)) // grad_accum_steps
    warmup_steps = max(1, total_steps // 10)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    ], milestones=[warmup_steps])

    for e in range(num_epochs):
        model.train()
        train_loss_accum = 0.0

        for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"TRAIN - E{e}"):
            x, y = x.to(device), y.to(device)
            x = add_spec_noise(x, p=0.1)

            model.detach_states()
            model.zero_states()

            # Dynamic timeline slicing
            frames_per_chunk = x.size(0) / 12.0
            step_losses = 0.0

            # For tracking batch ROC-AUC
            chunk_preds_train = torch.zeros_like(y)
            chunk_counts = torch.zeros(12, device=device)

            for t in range(x.size(0)):
                model_output = model(x[t])  # [B, num_classes]

                # Determine which 5s chunk we are currently in
                chunk_idx = min(int(t / frames_per_chunk), 11)

                target = y[:, chunk_idx, :]
                step_losses += loss_fn(model_output, target)

                # Accumulate for ROC-AUC
                chunk_preds_train[:, chunk_idx, :] += model_output
                chunk_counts[chunk_idx] += 1

            # Average loss over all 3750 steps
            loss = step_losses / x.size(0)
            (loss / grad_accum_steps).backward()

            accum_steps += 1
            train_loss_accum += loss.item()

            if accum_steps % grad_accum_steps == 0:
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], optimizer_steps)
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                update_ema_model(model, ema_model)
                optimizer.zero_grad()

                with torch.no_grad():
                    # Average logits within each chunk
                    for c in range(12):
                        if chunk_counts[c] > 0:
                            chunk_preds_train[:, c, :] /= chunk_counts[c]

                    train_probs = torch.sigmoid(chunk_preds_train).view(-1, num_classes).cpu().numpy()
                    train_targets = y.view(-1, num_classes).cpu().numpy()
                    train_roc_auc = compute_macro_roc_auc(train_probs, train_targets)

                writer.add_scalars("Loss", {"Train": train_loss_accum / grad_accum_steps}, optimizer_steps)
                writer.add_scalars("ROC_AUC", {"Train": train_roc_auc}, optimizer_steps)
                train_loss_accum = 0.0

            # ==========================================
            # VALIDATION LOOP (Every 200 steps)
            # ==========================================
            if optimizer_steps % 200 == 0 and optimizer_steps != 0:
                def evaluate_model(eval_model, name):
                    eval_model.eval()
                    val_loss = 0.0
                    all_preds, all_targets = [], []

                    with torch.no_grad():
                        for x_val, y_val in tqdm(val_dataloader, desc=f"VAL ({name})"):
                            x_val, y_val = x_val.to(device), y_val.to(device)
                            eval_model.zero_states()

                            frames_per_chunk_val = x_val.size(0) / 12.0
                            step_losses_val = 0.0

                            chunk_preds_val = torch.zeros_like(y_val)
                            chunk_counts_val = torch.zeros(12, device=device)

                            for t in range(x_val.size(0)):
                                model_output_val = eval_model(x_val[t])
                                chunk_idx = min(int(t / frames_per_chunk_val), 11)

                                target_val = y_val[:, chunk_idx, :]
                                step_losses_val += loss_fn(model_output_val, target_val)

                                chunk_preds_val[:, chunk_idx, :] += model_output_val
                                chunk_counts_val[chunk_idx] += 1

                            val_loss += (step_losses_val / x_val.size(0)).item()

                            for c in range(12):
                                if chunk_counts_val[c] > 0:
                                    chunk_preds_val[:, c, :] /= chunk_counts_val[c]

                            probs = torch.sigmoid(chunk_preds_val).view(-1, num_classes)

                            all_preds.append(probs.cpu().numpy())
                            all_targets.append(y_val.view(-1, num_classes).cpu().numpy())

                    if len(val_dataloader) > 0:
                        return val_loss / len(val_dataloader), compute_macro_roc_auc(np.concatenate(all_preds),
                                                                                     np.concatenate(all_targets))
                    return 0.0, 0.0


                print("\nEvaluating Base Model...")
                base_val_loss, base_roc_auc = evaluate_model(model, "Base")
                print("Evaluating EMA Model...")
                ema_val_loss, ema_roc_auc = evaluate_model(ema_model, "EMA")

                writer.add_scalars("Loss", {"Val_Base": base_val_loss, "Val_EMA": ema_val_loss}, optimizer_steps)
                writer.add_scalars("ROC_AUC", {"Val_Base": base_roc_auc, "Val_EMA": ema_roc_auc}, optimizer_steps)

                os.makedirs("checkpoints", exist_ok=True)
                save_file(ema_model.state_dict(),
                          os.path.join("checkpoints", f"birdclef_step_{optimizer_steps}_ema.safetensors"))
                save_file(model.state_dict(),
                          os.path.join("checkpoints", f"birdclef_step_{optimizer_steps}_base.safetensors"))

                print(f"\nSaved models. Base ROC-AUC: {base_roc_auc:.4f} | EMA ROC-AUC: {ema_roc_auc:.4f}\n")
                model.train()

    # END OF TRAINING SAVE
    save_file(ema_model.state_dict(), os.path.join("checkpoints", "birdclef_final_ema.safetensors"))
    save_file(model.state_dict(), os.path.join("checkpoints", "birdclef_final_base.safetensors"))
    writer.close()
