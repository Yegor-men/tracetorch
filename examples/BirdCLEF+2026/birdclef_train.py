import os
import copy
import socket
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

from birdclef_architecture import BirdClassifierSNN

# Fix the 'too many fds' multiprocessing error
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

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


class UnlabeledInferenceDataset(Dataset):
    def __init__(self, filenames, data_dir='data/birdclef-2026', sr=32000, duration=5.0):
        self.audio_dir = os.path.join(data_dir, "train_soundscapes")
        self.sr = sr
        self.duration = duration
        self.num_samples = int(duration * sr)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=2048, hop_length=512, n_mels=256, power=2.0)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.samples = [(f, idx) for f in filenames for idx in range(12)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, chunk_idx = self.samples[idx]
        file_path = os.path.join(self.audio_dir, filename)
        offset_sec = chunk_idx * self.duration

        try:
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
        return X, filename, chunk_idx


def collate_inference(batch):
    X, filenames, chunk_idxs = zip(*batch)
    return torch.stack(X).transpose(0, 1), filenames, chunk_idxs


class PseudoLabeledDataset(Dataset):
    def __init__(self, pseudo_labels_dict, data_dir='data/birdclef-2026', sr=32000, duration=5.0):
        self.audio_dir = os.path.join(data_dir, "train_soundscapes")
        self.sr = sr
        self.duration = duration
        self.num_samples = int(duration * sr)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=2048, hop_length=512, n_mels=256, power=2.0)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.samples = []
        for key, Y in pseudo_labels_dict.items():
            filename, chunk_idx = key.split('|')
            self.samples.append((filename, int(chunk_idx), Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, chunk_idx, Y = self.samples[idx]
        file_path = os.path.join(self.audio_dir, filename)
        offset_sec = chunk_idx * self.duration

        try:
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


def collate_student(batch):
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
            wf, sr = torchaudio.load(file_path)
            if sr != self.sr: wf = torchaudio.functional.resample(wf, sr, self.sr)
            if wf.shape[0] > 1: wf = wf.mean(dim=0, keepdim=True)

            # 60s cap
            waveform = wf[:, :self.num_samples]
            if waveform.shape[1] < self.num_samples:
                waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))
        except Exception:
            waveform = torch.zeros(1, self.num_samples)

        spec = self.mel_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0)
        spec_db = spec_db - spec_db.min()
        spec_db = spec_db / (spec_db.max() + 1e-6)

        X = spec_db.transpose(0, 1)
        return X, Y_matrix


def collate_labeled(batch):
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
    log_dir = f'tensorboard/birdclef_student_{timestamp}'
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

    ss_csv_path = os.path.join(data_dir, "train_soundscapes_labels.csv")
    df_ss_labels = pd.read_csv(ss_csv_path)
    labeled_files = set([f + '.ogg' if not f.endswith('.ogg') else f for f in df_ss_labels['filename'].unique()])

    all_files = set(os.listdir(os.path.join(data_dir, "train_soundscapes")))
    unlabeled_files = [f for f in all_files if f.endswith('.ogg') and f not in labeled_files]

    # =========================================================
    # PHASE 2: PSEUDO-LABEL GENERATION
    # =========================================================
    pseudo_labels_path = "pseudo_labels.pt"

    if os.path.exists(pseudo_labels_path):
        print(f"Loading existing Pseudo-Labels from {pseudo_labels_path}...")
        pseudo_labels_dict = torch.load(pseudo_labels_path)
    else:
        print("\n--- No pseudo_labels.pt found! Starting Teacher Inference Phase ---")

        teacher = BirdClassifierSNN(num_classes=num_classes).to(device)

        # !!! Ensure Teacher checkpoint is available from Phase 1
        teacher.load_state_dict(load_file("checkpoints/teacher_final_ema.safetensors"))
        teacher.eval()

        inference_ds = UnlabeledInferenceDataset(unlabeled_files, data_dir=data_dir)
        inference_loader = DataLoader(inference_ds, batch_size=64, shuffle=False, collate_fn=collate_inference,
                                      num_workers=4)

        pseudo_labels_dict = {}

        with torch.no_grad():
            for X, batch_filenames, batch_chunks in tqdm(inference_loader, desc="Pseudo-Labeling"):
                X = X.to(device)
                teacher.zero_states()

                all_step_logits = []
                for t in range(X.size(0)):
                    all_step_logits.append(teacher(X[t]))

                mean_logits = torch.stack(all_step_logits).mean(dim=0)

                # Binarize outputs: if mean logits > 0, model is confident it's present
                binary_preds = (mean_logits > 0.0).float().cpu()

                for b in range(len(batch_filenames)):
                    key = f"{batch_filenames[b]}|{batch_chunks[b]}"
                    pseudo_labels_dict[key] = binary_preds[b]

        torch.save(pseudo_labels_dict, pseudo_labels_path)
        print(f"Saved {len(pseudo_labels_dict)} pseudo-labels to {pseudo_labels_path}\n")

    # =========================================================
    # PHASE 3: STUDENT TRAINING
    # =========================================================
    train_ds = PseudoLabeledDataset(pseudo_labels_dict, data_dir=data_dir)
    val_ds = LabeledSoundscapeDataset(df_ss_labels, class_to_idx, data_dir=data_dir)

    print(f"Training Student on {len(train_ds)} pseudo-labeled 5s crops.")
    print(f"Validating Student on {len(val_ds)} cleanly labeled 60s soundscapes.")

    batch_size = 32
    grad_accum_steps = 2

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_student,
                                  num_workers=4)
    val_dataloader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_labeled, num_workers=4)

    student_model = BirdClassifierSNN(num_classes=num_classes).to(device)
    ema_student_model = copy.deepcopy(student_model)
    ema_student_model.eval()
    for param in ema_student_model.parameters(): param.requires_grad = False

    # ==========================================
    # SANITY CHECKS & PARAMETER LOGGING
    # ==========================================
    total_params = sum(p.numel() for p in student_model.parameters())
    snn_params = student_model.get_param_count()
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

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer_steps, accum_steps, num_epochs = 0, 0, 5
    total_steps = num_epochs * max(1, len(train_dataloader)) // grad_accum_steps
    warmup_steps = max(1, total_steps // 10)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    ], milestones=[warmup_steps])

    for e in range(num_epochs):
        student_model.train()
        train_loss_accum = 0.0

        for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"STUDENT TRAIN - E{e}"):
            x, y = x.to(device), y.to(device)
            x = add_spec_noise(x, p=0.1)

            student_model.zero_states()
            all_step_logits = []

            for t in range(x.size(0)):
                logits = student_model(x[t])
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

                update_ema_model(student_model, ema_student_model)
                optimizer.zero_grad()

                writer.add_scalars("Loss", {"Train_BCE": train_loss_accum / grad_accum_steps}, optimizer_steps)
                train_loss_accum = 0.0

                if optimizer_steps % 500 == 0:
                    def evaluate_student(model, name):
                        model.eval()
                        all_preds, all_targets = [], []

                        with torch.no_grad():
                            for x_val, y_val in tqdm(val_dataloader, desc=f"VAL ({name})", leave=False):
                                x_val, y_val = x_val.to(device), y_val.to(device)
                                model.zero_states()

                                frames_per_chunk_val = x_val.size(0) / 12.0
                                chunk_logits_sum = torch.zeros_like(y_val)
                                chunk_counts = torch.zeros(12, device=device)

                                for t in range(x_val.size(0)):
                                    logits = model(x_val[t])
                                    chunk_idx = min(int(t / frames_per_chunk_val), 11)
                                    chunk_logits_sum[:, chunk_idx, :] += logits
                                    chunk_counts[chunk_idx] += 1

                                for c in range(12):
                                    if chunk_counts[c] > 0:
                                        chunk_logits_sum[:, c, :] /= chunk_counts[c]

                                probs = torch.sigmoid(chunk_logits_sum).view(-1, num_classes)
                                all_preds.append(probs.cpu().numpy())
                                all_targets.append(y_val.view(-1, num_classes).cpu().numpy())

                        if len(val_dataloader) > 0:
                            return compute_macro_roc_auc(np.concatenate(all_preds), np.concatenate(all_targets))
                        return 0.0


                    print("\nEvaluating Base Student...")
                    base_roc_auc = evaluate_student(student_model, "Base")
                    print("Evaluating EMA Student...")
                    ema_roc_auc = evaluate_student(ema_student_model, "EMA")

                    writer.add_scalars("ROC_AUC", {"Val_Base": base_roc_auc, "Val_EMA": ema_roc_auc}, optimizer_steps)

                    os.makedirs("checkpoints", exist_ok=True)
                    save_file(ema_student_model.state_dict(),
                              os.path.join("checkpoints", f"student_step_{optimizer_steps}_ema.safetensors"))
                    print(f"\nSaved student. Base ROC-AUC: {base_roc_auc:.4f} | EMA ROC-AUC: {ema_roc_auc:.4f}\n")

                    student_model.train()

    save_file(ema_student_model.state_dict(), os.path.join("checkpoints", "student_final_ema.safetensors"))
