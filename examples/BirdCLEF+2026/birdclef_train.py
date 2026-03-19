import os
import ast
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
from safetensors.torch import save_file, load_file
import librosa
from sklearn.metrics import roc_auc_score
import subprocess
import webbrowser
import time
import atexit

from birdclef_architecture import SNNWorldModel, ClassificationDecoder

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.manual_seed_all(0)


def compute_macro_roc_auc(preds, targets):
    binary_targets = (targets > 0.0).astype(np.float32)
    target_sums = binary_targets.sum(axis=0)

    scored_columns = np.where((target_sums > 0) & (target_sums < len(targets)))[0]
    if len(scored_columns) == 0: return 0.5

    return roc_auc_score(binary_targets[:, scored_columns], preds[:, scored_columns], average='macro')


def add_spec_noise(tensor: torch.Tensor, p: float = 0.1, device=None) -> torch.Tensor:
    if p <= 0: return tensor
    if device is None: device = tensor.device
    mask = torch.rand(tensor.shape, device=device) > p
    return tensor * mask


@torch.no_grad()
def update_ema_model(model, ema_model, decay: float = 0.999):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


class CleanAudioDataset(Dataset):
    def __init__(self, df, class_to_idx, data_dir='data/birdclef-2026', sr=32000, duration=5.0):
        self.df = df
        self.class_to_idx = class_to_idx
        self.audio_dir = os.path.join(data_dir, "train_audio")
        self.sr = sr
        self.target_length = int(duration * sr)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=2048, hop_length=512, n_mels=256, power=2.0)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.class_files_map = self.df.groupby('primary_label')['filename'].apply(list).to_dict()

    def _safe_load(self, filename):
        file_path = os.path.join(self.audio_dir, filename if filename.endswith('.ogg') else filename + '.ogg')
        try:
            wf, _ = librosa.load(file_path, sr=self.sr, mono=True)
            return torch.from_numpy(wf)
        except Exception:
            return torch.zeros(self.sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        primary_label = row['primary_label']
        filename = str(row['filename'])

        Y = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        if primary_label in self.class_to_idx:
            Y[self.class_to_idx[primary_label]] = 1.0

        sec_labels = row.get('secondary_labels', '[]')
        if isinstance(sec_labels, str):
            try:
                sec_labels = ast.literal_eval(sec_labels)
            except Exception:
                sec_labels = []
        if isinstance(sec_labels, list):
            for sl in sec_labels:
                if sl in self.class_to_idx: Y[self.class_to_idx[sl]] = 1.0

        waveform_parts = []
        current_length = 0

        wf = self._safe_load(filename)
        waveform_parts.append(wf)
        current_length += wf.shape[0]

        available_files = self.class_files_map.get(primary_label, [filename])
        while current_length < self.target_length:
            next_filename = random.choice(available_files)
            wf = self._safe_load(next_filename)
            waveform_parts.append(wf)
            current_length += wf.shape[0]

        waveform = torch.cat(waveform_parts, dim=0)

        max_start = current_length - self.target_length
        start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
        waveform = waveform[start:start + self.target_length].unsqueeze(0)

        spec = self.mel_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0)
        X = ((spec_db + 80.0) / 80.0).transpose(0, 1)

        return X, Y


def collate_clean(batch):
    X, Y = zip(*batch)
    return torch.stack(X).transpose(0, 1), torch.stack(Y)


class LabeledSoundscapeDataset(Dataset):
    def __init__(self, df_labels, class_to_idx, data_dir='data/birdclef-2026', sr=32000):
        self.sr = sr
        self.num_samples = int(60.0 * sr)
        self.audio_dir = os.path.join(data_dir, "train_soundscapes")
        self.class_to_idx = class_to_idx

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=2048, hop_length=512, n_mels=256, power=2.0)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.samples = []
        unique_files = df_labels['filename'].unique()

        for filename in unique_files:
            file_path = os.path.join(self.audio_dir, str(filename))
            if not file_path.endswith('.ogg'): file_path += '.ogg'

            Y_matrix = torch.zeros((12, len(self.class_to_idx)), dtype=torch.float32)

            file_df = df_labels[df_labels['filename'] == filename]
            for _, row in file_df.iterrows():
                time_parts = str(row['start']).split(':')
                start_sec = sum(float(x) * (60 ** i) for i, x in enumerate(reversed(time_parts)))
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
            waveform_np, _ = librosa.load(file_path, sr=self.sr, mono=True, duration=60.0)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        except Exception:
            waveform = torch.zeros(1, self.num_samples)

        if waveform.shape[1] < self.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))

        spec = self.mel_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0)
        X = ((spec_db + 80.0) / 80.0).transpose(0, 1)

        return X, Y_matrix


def collate_labeled(batch):
    X, Y = zip(*batch)
    return torch.stack(X).transpose(0, 1), torch.stack(Y)


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'tensorboard/birdclef_class_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)

    subprocess.Popen(['tensorboard', '--logdir', "tensorboard", '--port', '6006'])
    time.sleep(2)
    webbrowser.open('http://localhost:6006')


    def cleanup():
        writer.flush()
        writer.close()


    atexit.register(cleanup)

    data_dir = 'data/birdclef-2026'

    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    label_col = 'primary_label' if 'primary_label' in df_train.columns else df_train.columns[1]
    classes = sorted(df_train[label_col].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    train_ds = CleanAudioDataset(df_train, class_to_idx, data_dir=data_dir, duration=5.0)

    df_ss_labels = pd.read_csv(os.path.join(data_dir, "train_soundscapes_labels.csv"))
    val_ds = LabeledSoundscapeDataset(df_ss_labels, class_to_idx, data_dir=data_dir)

    print(f"Training on {len(train_ds)} organically concatenated 5s crops.")
    print(f"Validating on {len(val_ds)} messy labeled 60s soundscapes.")

    batch_size = 50
    grad_accum_steps = 1

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_clean,
                                  num_workers=4)
    val_dataloader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_labeled, num_workers=4)

    # ==========================================
    # SANITY CHECKS & PARAMETER LOGGING
    # ==========================================
    world_model = SNNWorldModel().to(device)
    world_model.load_state_dict(load_file("checkpoints/world_model_step_500_ema.safetensors"))
    world_model.eval()
    for param in world_model.parameters(): param.requires_grad = False

    class_decoder = ClassificationDecoder(latent_dim=1024, num_classes=num_classes).to(device)
    ema_class_decoder = copy.deepcopy(class_decoder)
    ema_class_decoder.eval()
    for param in ema_class_decoder.parameters(): param.requires_grad = False

    total_params = sum(p.numel() for p in world_model.parameters()) + sum(p.numel() for p in class_decoder.parameters())
    snn_params = world_model.get_param_count()
    print(f"\nTotal Params: {total_params:,} -> SNN: {snn_params:,} | Non-SNN: {total_params - snn_params:,}")
    print(f"Num Train Batches: {len(train_dataloader):,}\n")

    # Fetch 1 batch to verify shape and render spectrogram
    x_sanity, y_sanity = next(iter(train_dataloader))
    print(f"Sanity Check - X shape (Time, Batch, Freq): {x_sanity.shape} | Y shape: {y_sanity.shape}")

    # Fix orientation: [Time, Batch, Freq] -> [Batch, Freq, Time] -> [Batch, 1, Freq, Time]
    x_vis = x_sanity.permute(1, 2, 0).unsqueeze(1)
    import tracetorch as tt

    tt.plot.render_image(x_vis)
    print("Rendered Sanity Check Image.\n")
    # ==========================================

    optimizer = torch.optim.AdamW(class_decoder.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer_steps, accum_steps, num_epochs = 0, 0, 5
    total_steps = num_epochs * max(1, len(train_dataloader)) // grad_accum_steps
    warmup_steps = max(1, total_steps // 10)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    ], milestones=[warmup_steps])

    for e in range(num_epochs):
        class_decoder.train()
        train_loss_accum = 0.0

        for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"TRAIN - E{e}"):
            x, y = x.to(device), y.to(device)
            x = add_spec_noise(x, p=0.1)

            world_model.zero_states()

            all_step_logits = []

            for t in range(x.size(0)):
                with torch.no_grad():
                    latents = world_model(x[t])

                logits = class_decoder(latents)
                all_step_logits.append(logits)

            # [Time, Batch, num_classes]
            all_step_logits = torch.stack(all_step_logits)

            # Temporal Pooling over the exact 5s window.
            # Using mean is mathematically identical to a linear sum but keeps values bounded for BCEWithLogitsLoss
            clip_logits = torch.mean(all_step_logits, dim=0)

            loss = loss_fn(clip_logits, y)
            (loss / grad_accum_steps).backward()

            accum_steps += 1
            train_loss_accum += loss.item()

            if accum_steps % grad_accum_steps == 0:
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], optimizer_steps)
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1

                update_ema_model(class_decoder, ema_class_decoder)
                optimizer.zero_grad()

                writer.add_scalars("Loss", {"Train_BCE": train_loss_accum / grad_accum_steps}, optimizer_steps)
                train_loss_accum = 0.0

                if optimizer_steps % 500 == 0:
                    def evaluate_decoder(decoder, name):
                        decoder.eval()
                        all_preds, all_targets = [], []

                        with torch.no_grad():
                            for x_val, y_val in tqdm(val_dataloader, desc=f"VAL ({name})", leave=False):
                                x_val, y_val = x_val.to(device), y_val.to(device)
                                world_model.zero_states()

                                frames_per_chunk_val = x_val.size(0) / 12.0

                                chunk_logits_sum = torch.zeros_like(y_val)
                                chunk_counts = torch.zeros(12, device=device)

                                for t in range(x_val.size(0)):
                                    logits = decoder(world_model(x_val[t]))
                                    chunk_idx = min(int(t / frames_per_chunk_val), 11)
                                    chunk_logits_sum[:, chunk_idx, :] += logits
                                    chunk_counts[chunk_idx] += 1

                                for c in range(12):
                                    if chunk_counts[c] > 0:
                                        # Mean pool across each specific 5s soundscape chunk
                                        chunk_logits_sum[:, c, :] /= chunk_counts[c]

                                probs = torch.sigmoid(chunk_logits_sum).view(-1, num_classes)
                                all_preds.append(probs.cpu().numpy())
                                all_targets.append(y_val.view(-1, num_classes).cpu().numpy())

                        if len(val_dataloader) > 0:
                            return compute_macro_roc_auc(np.concatenate(all_preds), np.concatenate(all_targets))
                        return 0.0


                    print("\nEvaluating Base Decoder...")
                    base_roc_auc = evaluate_decoder(class_decoder, "Base")
                    print("Evaluating EMA Decoder...")
                    ema_roc_auc = evaluate_decoder(ema_class_decoder, "EMA")

                    writer.add_scalars("ROC_AUC", {"Val_Base": base_roc_auc, "Val_EMA": ema_roc_auc}, optimizer_steps)

                    os.makedirs("checkpoints", exist_ok=True)
                    save_file(ema_class_decoder.state_dict(),
                              os.path.join("checkpoints", f"class_decoder_step_{optimizer_steps}_ema.safetensors"))
                    print(f"\nSaved decoders. Base ROC-AUC: {base_roc_auc:.4f} | EMA ROC-AUC: {ema_roc_auc:.4f}\n")

                    class_decoder.train()

    save_file(ema_class_decoder.state_dict(), os.path.join("checkpoints", "class_decoder_final_ema.safetensors"))
