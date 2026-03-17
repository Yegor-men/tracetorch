import os
import copy
import random
import ast
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
from safetensors.torch import save_file, load_file
import librosa
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

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
    """
    Exact implementation of the Kaggle BirdCLEF 2026 ROC-AUC metric.
    Only scores classes that have >0 true positive labels in the given targets.
    """
    target_sums = targets.sum(axis=0)
    scored_columns = np.where(target_sums > 0)[0]

    if len(scored_columns) == 0:
        return 0.5  # If absolutely no positive targets exist, return random guessing baseline

    return roc_auc_score(targets[:, scored_columns], preds[:, scored_columns], average='macro')


# =====================================================================
# 1. Dataset & Dataloader
# =====================================================================
class BirdCLEFDataset(Dataset):
    def __init__(self, data_dir='data/birdclef-2026', split='train', duration=5.0, sr=32000):
        self.data_dir = data_dir
        self.sr = sr
        self.num_samples = int(duration * sr)

        self.audio_dir = os.path.join(data_dir, "train_audio")
        self.csv_path = os.path.join(data_dir, "train.csv")

        if not os.path.exists(self.csv_path) or not os.path.exists(self.audio_dir):
            raise FileNotFoundError(
                f"\n[ERROR] Could not find the dataset at {data_dir}.\n"
                f"Please ensure:\n"
                f"1. {self.csv_path} exists.\n"
                f"2. {self.audio_dir}/ exists."
            )

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=2048,
            hop_length=512,
            n_mels=256,
            power=2.0
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.samples = []
        df_full = pd.read_csv(self.csv_path)

        filename_col = 'filename' if 'filename' in df_full.columns else df_full.columns[0]
        label_col = 'primary_label' if 'primary_label' in df_full.columns else df_full.columns[1]

        self.classes = sorted(df_full[label_col].astype(str).unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if split == 'train':
            df = df_full.sample(frac=0.8, random_state=42)
        else:
            df = df_full.drop(df_full.sample(frac=0.8, random_state=42).index)

        for _, row in df.iterrows():
            file_path = os.path.join(self.audio_dir, str(row[filename_col]))
            if not file_path.endswith('.ogg'):
                file_path += '.ogg'

            # Parse primary and secondary labels for Multi-Hot Encoding
            primary = str(row[label_col])
            secondaries = []
            if 'secondary_labels' in row:
                try:
                    parsed = ast.literal_eval(row['secondary_labels'])
                    if isinstance(parsed, list):
                        secondaries = [str(lbl) for lbl in parsed]
                except Exception:
                    pass

            label_indices = [self.class_to_idx[primary]]
            for sec in secondaries:
                if sec in self.class_to_idx:
                    label_indices.append(self.class_to_idx[sec])

            self.samples.append((file_path, label_indices))

        print(f"Loaded {split} split: {len(self.samples)} real audio samples spanning {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label_indices = self.samples[idx]

        try:
            waveform_np, _ = librosa.load(file_path, sr=self.sr, mono=True)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        except Exception as e:
            if not hasattr(self, '_printed_error'):
                print(f"\n[CRITICAL ERROR] Failed to load audio!")
                self._printed_error = True
            waveform = torch.zeros(1, self.num_samples)

        if waveform.shape[1] > self.num_samples:
            start = random.randint(0, waveform.shape[1] - self.num_samples)
            waveform = waveform[:, start:start + self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            pad = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        spec = self.mel_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0)

        spec_current = (spec_db + 80.0) / 80.0

        X = spec_current.transpose(0, 1)  # [L, E] -> [313, 256]

        Y = torch.zeros(len(self.classes), dtype=torch.float32)
        Y[label_indices] = 1.0

        return X, Y


def collate_fn(batch):
    X, Y = zip(*batch)
    X = torch.stack(X).transpose(0, 1)  # ->[L, B, 256]
    Y = torch.stack(Y)  # ->[B, num_classes] (float32)
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

        self.enc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Dropout(0.0),
        )

        layers = [ResidualSpike(hidden_features) for _ in range(num_layers)]
        self.net = nn.Sequential(*layers)

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
    import subprocess
    import webbrowser
    import time
    from torch.utils.tensorboard import SummaryWriter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'tensorboard/birdclef_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)

    subprocess.Popen(['tensorboard', '--logdir', 'tensorboard', '--port', '6006'],
                     stderr=subprocess.DEVNULL)
    time.sleep(2)
    webbrowser.open('http://localhost:6006')

    batch_size = 60
    minibatch_size = 1

    train_ds = BirdCLEFDataset(split='train')
    val_ds = BirdCLEFDataset(split='val')
    num_classes = len(train_ds.classes)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = SNN(in_features=256, hidden_features=1024, num_layers=10, out_features=num_classes).to(device)
    model.load_state_dict(load_file("checkpoints/_birdclef_step_1600_ema.safetensors"))

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), 1e-5)

    # Reverting to raw, unweighted BCE for pure calibration
    loss_fn = nn.BCEWithLogitsLoss()

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
        train_loss_accum = 0.0

        for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"TRAIN - E{e}"):
            x = x.to(device)  # [313, B, 256]
            y = y.to(device)  # [B, num_classes]

            x = add_spec_noise(x, p=0.1)

            model.detach_states()
            model.zero_states()

            # Pass over time, keep only final output
            for t in range(x.size(0)):
                model_output = model(x[t])

            loss = loss_fn(model_output, y)
            scaled_loss = loss / minibatch_size
            scaled_loss.backward()

            accum_steps += 1
            train_loss_accum += loss.item()

            if accum_steps % minibatch_size == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("learning_rate", current_lr, optimizer_steps)

                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                update_ema_model(model, ema_model)
                optimizer.zero_grad()

                # Calculate Training ROC-AUC for this specific batch
                with torch.no_grad():
                    train_probs = torch.sigmoid(model_output).cpu().numpy()
                    train_targets = y.cpu().numpy()
                    train_roc_auc = compute_macro_roc_auc(train_probs, train_targets)

                avg_train_loss = train_loss_accum / minibatch_size

                # Log to the shared main tags so they appear on the same graphs
                writer.add_scalars("Loss", {"Train": avg_train_loss}, optimizer_steps)
                writer.add_scalars("ROC_AUC", {"Train": train_roc_auc}, optimizer_steps)

                train_loss_accum = 0.0

            # ==========================================
            # VALIDATION LOOP (Every 200 steps)
            # ==========================================
            if optimizer_steps % 200 == 0 and optimizer_steps != 0:

                def evaluate_model(eval_model, name):
                    eval_model.eval()
                    val_loss = 0.0
                    all_preds = []
                    all_targets = []

                    with torch.no_grad():
                        for j, (x_val, y_val) in tqdm(enumerate(val_dataloader), total=len(val_dataloader),
                                                      desc=f"VAL ({name})"):
                            x_val = x_val.to(device)
                            y_val = y_val.to(device)
                            eval_model.zero_states()

                            for t in range(x_val.size(0)):
                                model_output_val = eval_model(x_val[t])

                            loss_val = loss_fn(model_output_val, y_val)
                            val_loss += loss_val.item()

                            probs = torch.sigmoid(model_output_val)

                            all_preds.append(probs.cpu().numpy())
                            all_targets.append(y_val.cpu().numpy())

                    if len(val_dataloader) > 0:
                        val_loss /= len(val_dataloader)
                        all_preds = np.concatenate(all_preds, axis=0)
                        all_targets = np.concatenate(all_targets, axis=0)

                        roc_auc = compute_macro_roc_auc(all_preds, all_targets)
                        return val_loss, roc_auc
                    return 0.0, 0.0


                print("\nEvaluating Base Model...")
                base_val_loss, base_roc_auc = evaluate_model(model, "Base")
                print("Evaluating EMA Model...")
                ema_val_loss, ema_roc_auc = evaluate_model(ema_model, "EMA")

                # Log to the same shared main tags
                writer.add_scalars("Loss", {"Val_Base": base_val_loss, "Val_EMA": ema_val_loss}, optimizer_steps)
                writer.add_scalars("ROC_AUC", {"Val_Base": base_roc_auc, "Val_EMA": ema_roc_auc}, optimizer_steps)

                os.makedirs("checkpoints", exist_ok=True)
                ema_path = os.path.join("checkpoints", f"birdclef_step_{optimizer_steps}_ema.safetensors")
                base_path = os.path.join("checkpoints", f"birdclef_step_{optimizer_steps}_base.safetensors")
                save_file(ema_model.state_dict(), ema_path)
                save_file(model.state_dict(), base_path)

                print(f"\nSaved models. Base ROC-AUC: {base_roc_auc:.4f} | EMA ROC-AUC: {ema_roc_auc:.4f}\n")

                # Explicitly return base model to train mode
                model.train()

    # ==========================================
    # END OF TRAINING SAVE
    # ==========================================
    os.makedirs("checkpoints", exist_ok=True)
    ema_path_final = os.path.join("checkpoints", "birdclef_final_ema.safetensors")
    base_path_final = os.path.join("checkpoints", "birdclef_final_base.safetensors")
    save_file(ema_model.state_dict(), ema_path_final)
    save_file(model.state_dict(), base_path_final)
    print(f"\nTraining Complete! Saved final models to {ema_path_final} and {base_path_final}")

    writer.close()
