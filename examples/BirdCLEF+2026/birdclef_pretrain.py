import os
import ast
import copy
import random
import socket
import pandas as pd
import numpy as np
import torchaudio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from safetensors.torch import save_file
import subprocess
import webbrowser
import time
import atexit

from birdclef_architecture import BirdClassifierSNN

# Fix the 'too many fds' multiprocessing error
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.manual_seed_all(0)


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
            wf, sr = torchaudio.load(file_path)
            if sr != self.sr:
                wf = torchaudio.functional.resample(wf, sr, self.sr)
            if wf.shape[0] > 1:
                wf = wf.mean(dim=0, keepdim=True)
            return wf.squeeze(0)
        except Exception:
            return torch.zeros(self.sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        primary_label = row['primary_label']
        filename = str(row['filename'])

        Y = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        if primary_label in self.class_to_idx: Y[self.class_to_idx[primary_label]] = 1.0

        sec_labels = row.get('secondary_labels', '[]')
        if isinstance(sec_labels, str):
            try:
                sec_labels = ast.literal_eval(sec_labels)
            except Exception:
                sec_labels = []
        if isinstance(sec_labels, list):
            for sl in sec_labels:
                if sl in self.class_to_idx: Y[self.class_to_idx[sl]] = 1.0

        # Organically stack the same bird's recordings to strictly eliminate silence
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

        # Strict Min-Max Normalization to ensure [0.0, 1.0] range
        spec_db = spec_db - spec_db.min()
        spec_db = spec_db / (spec_db.max() + 1e-6)

        X = spec_db.transpose(0, 1)  # [Time, Freq]
        return X, Y


class LabeledChunkDataset(Dataset):
    def __init__(self, df_labels, class_to_idx, data_dir='data/birdclef-2026', sr=32000, duration=5.0):
        self.sr = sr
        self.duration = duration
        self.num_samples = int(duration * sr)
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

            file_df = df_labels[df_labels['filename'] == filename]
            for chunk_idx in range(12):
                Y = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
                chunk_df = file_df[file_df['start'].apply(lambda x: int(
                    sum(float(p) * (60 ** i) for i, p in enumerate(reversed(str(x).split(':')))) // 5) == chunk_idx)]

                for _, row in chunk_df.iterrows():
                    birds_str = str(row['primary_label'])
                    birds = birds_str.split() if ' ' in birds_str else [birds_str]
                    for b in birds:
                        if b in self.class_to_idx: Y[self.class_to_idx[b]] = 1.0

                self.samples.append((file_path, chunk_idx, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, chunk_idx, Y = self.samples[idx]
        offset_sec = chunk_idx * self.duration
        try:
            # Load whole file and slice in memory (faster/safer than seeking)
            wf, sr = torchaudio.load(file_path)
            if sr != self.sr: wf = torchaudio.functional.resample(wf, sr, self.sr)
            if wf.shape[0] > 1: wf = wf.mean(dim=0, keepdim=True)

            start_sample = int(offset_sec * self.sr)
            end_sample = start_sample + self.num_samples
            waveform = wf[:, start_sample:end_sample]

            if waveform.shape[1] < self.num_samples:
                waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))
        except Exception:
            waveform = torch.zeros(1, self.num_samples)

        spec = self.mel_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0)

        spec_db = spec_db - spec_db.min()
        spec_db = spec_db / (spec_db.max() + 1e-6)

        X = spec_db.transpose(0, 1)
        return X, Y


def collate_combined(batch):
    X, Y = zip(*batch)
    return torch.stack(X).transpose(0, 1), torch.stack(Y)


def find_free_port(start_port=6006):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0: return port
        port += 1


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'tensorboard/birdclef_teacher_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)

    tb_port = find_free_port(6006)
    subprocess.Popen(['tensorboard', '--logdir', "tensorboard", '--port', str(tb_port)], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    time.sleep(2)
    webbrowser.open(f'http://localhost:{tb_port}')


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

    clean_ds = CleanAudioDataset(df_train, class_to_idx, data_dir=data_dir, duration=5.0)

    df_ss_labels = pd.read_csv(os.path.join(data_dir, "train_soundscapes_labels.csv"))
    labeled_chunk_ds = LabeledChunkDataset(df_ss_labels, class_to_idx, data_dir=data_dir, duration=5.0)

    combined_ds = ConcatDataset([clean_ds, labeled_chunk_ds])

    print(
        f"Training Teacher on {len(clean_ds)} clean 5s crops + {len(labeled_chunk_ds)} labeled soundscape chunks = {len(combined_ds)} Total.")

    batch_size = 50
    grad_accum_steps = 1

    train_dataloader = DataLoader(combined_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_combined,
                                  num_workers=4)

    # ==========================================
    # SANITY CHECKS & PARAMETER LOGGING
    # ==========================================
    teacher_model = BirdClassifierSNN(num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in teacher_model.parameters())
    snn_params = teacher_model.get_param_count()
    print(f"\nTotal Params: {total_params:,} -> SNN: {snn_params:,} | Non-SNN: {total_params - snn_params:,}")
    print(f"Num Train Batches: {len(train_dataloader):,}\n")

    x_sanity, y_sanity = next(iter(train_dataloader))
    print(f"Sanity Check - X shape (Time, Batch, Freq): {x_sanity.shape} | Y shape: {y_sanity.shape}")
    print(f"Sanity Check - X Mean: {x_sanity.mean():.4f} | Min: {x_sanity.min():.4f} | Max: {x_sanity.max():.4f}")

    x_vis = x_sanity.permute(1, 2, 0).unsqueeze(1)
    import tracetorch as tt

    tt.plot.render_image(x_vis)
    print("Rendered Sanity Check Image.\n")
    # ==========================================

    ema_teacher_model = copy.deepcopy(teacher_model)
    ema_teacher_model.eval()
    for param in ema_teacher_model.parameters(): param.requires_grad = False

    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer_steps, accum_steps, num_epochs = 0, 0, 5
    total_steps = num_epochs * max(1, len(train_dataloader)) // grad_accum_steps
    warmup_steps = max(1, total_steps // 10)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    ], milestones=[warmup_steps])

    for e in range(num_epochs):
        teacher_model.train()
        train_loss_accum = 0.0

        for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                              desc=f"TEACHER PRETRAIN - E{e}"):
            x, y = x.to(device), y.to(device)
            x = add_spec_noise(x, p=0.1)

            teacher_model.zero_states()
            all_step_logits = []

            for t in range(x.size(0)):
                logits = teacher_model(x[t])
                all_step_logits.append(logits)

            all_step_logits = torch.stack(all_step_logits)
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

                update_ema_model(teacher_model, ema_teacher_model)
                optimizer.zero_grad()

                writer.add_scalars("Loss", {"Train_BCE": train_loss_accum / grad_accum_steps}, optimizer_steps)
                train_loss_accum = 0.0

                if optimizer_steps % 500 == 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    save_file(ema_teacher_model.state_dict(),
                              os.path.join("checkpoints", f"teacher_step_{optimizer_steps}_ema.safetensors"))
                    print(f"\nSaved Teacher Model at step {optimizer_steps}\n")

    save_file(ema_teacher_model.state_dict(), os.path.join("checkpoints", "teacher_final_ema.safetensors"))
