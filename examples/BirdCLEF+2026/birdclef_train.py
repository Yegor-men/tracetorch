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

        # Map class names to all their respective files to enable organic concatenation
        self.class_files_map = self.df.groupby('primary_label')['filename'].apply(list).to_dict()

    def _safe_load(self, filename):
        file_path = os.path.join(self.audio_dir, filename if filename.endswith('.ogg') else filename + '.ogg')
        try:
            wf, _ = librosa.load(file_path, sr=self.sr, mono=True)
            return torch.from_numpy(wf)
        except Exception:
            # Fallback zero tensor so process doesn't completely die on bad files
            return torch.zeros(self.sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        primary_label = row['primary_label']
        filename = str(row['filename'])

        # Establish Target labels (Primary + Secondary)
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

        # Organic audio concatenation to avoid zero-padding
        waveform_parts = []
        current_length = 0

        # Pull original file
        wf = self._safe_load(filename)
        waveform_parts.append(wf)
        current_length += wf.shape[0]

        # Keep appending random files of the SAME species until we satisfy the 5s requirement
        available_files = self.class_files_map.get(primary_label, [filename])
        while current_length < self.target_length:
            next_filename = random.choice(available_files)
            wf = self._safe_load(next_filename)
            waveform_parts.append(wf)
            current_length += wf.shape[0]

        waveform = torch.cat(waveform_parts, dim=0)

        # Slice perfectly down to exactly 5s randomly
        max_start = current_length - self.target_length
        start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
        waveform = waveform[start:start + self.target_length].unsqueeze(0)

        # Transform to Spectrogram
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

    # 1. Setup Data
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

    batch_size = 16
