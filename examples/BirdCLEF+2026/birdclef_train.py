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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from safetensors.torch import save_file
import subprocess
import webbrowser
import time
import atexit
from sklearn.metrics import roc_auc_score

import tracetorch as tt
from birdclef_architecture import BirdClassifierSNN

# Fix the 'too many fds' multiprocessing error
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.manual_seed_all(0)


# =====================================================================
# 1. UTILS & CUSTOM LOSS
# =====================================================================
@torch.no_grad()
def update_ema_model(model, ema_model, decay: float = 0.999):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def compute_macro_roc_auc(preds, targets):
    binary_targets = (targets > 0.0).astype(np.float32)
    target_sums = binary_targets.sum(axis=0)

    scored_columns = np.where((target_sums > 0) & (target_sums < len(targets)))[0]
    if len(scored_columns) == 0: return 0.5
    return roc_auc_score(binary_targets[:, scored_columns], preds[:, scored_columns], average='macro')


def custom_multi_class_ce_loss(logits, targets):
    lsm = torch.nn.functional.log_softmax(logits, dim=1)
    sum_lsm_positives = (targets * lsm).sum(dim=1)
    num_positives = torch.clamp(targets.sum(dim=1), min=1.0)
    loss_per_sample = -sum_lsm_positives / num_positives
    return loss_per_sample.mean()


def focal_loss_with_logits(logits, targets, gamma=2.0):
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce_loss).mean()


def find_free_port(start_port=6006):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0: return port
        port += 1


# =====================================================================
# 2. FEATURE EXTRACTION & EVALUATION
# =====================================================================
def extract_features(x_spec, delta_fn, training=False, noise_std=0.01):
    """
    Takes a raw spectrogram batch[Time, Batch, Freq], applies optional noise,
    and returns the concatenated [Spec, Delta, Accel] tensor.
    """
    if training and noise_std > 0.0:
        x_spec = x_spec + torch.randn_like(x_spec) * noise_std

    # ComputeDeltas expects[..., Freq, Time], so we permute
    x_perm = x_spec.permute(1, 2, 0)

    delta = delta_fn(x_perm)
    delta_delta = delta_fn(delta)

    # Variance Matching
    delta_scalar = 1.0 / (0.1 ** 0.5)
    accel_scalar = 1.0 / 0.1

    delta = delta * delta_scalar
    delta_delta = delta_delta * accel_scalar

    # Concatenate into 3 channels * 256 = 768 Features
    combined = torch.cat([x_perm, delta, delta_delta], dim=1)

    # Return to [Time, Batch, Freq]
    return combined.permute(2, 0, 1)


def evaluate_model(eval_model, val_loader, bce_weight, delta_fn):
    eval_model.eval()
    all_preds, all_targets = [], []
    val_loss_sum, val_ce_sum, val_bce_sum = 0.0, 0.0, 0.0

    with torch.no_grad():
        for x_val, y_val in tqdm(val_loader, desc="Evaluating", leave=False):
            x_val, y_val = x_val.to(device), y_val.to(device)

            # Extract features WITHOUT noise for validation
            x_feat = extract_features(x_val, delta_fn, training=False)

            eval_model.zero_states()

            val_step_logits = []
            for t in range(x_feat.size(0)):
                val_step_logits.append(eval_model(x_feat[t]))

            val_clip_logits = torch.stack(val_step_logits).mean(dim=0)

            val_ce = custom_multi_class_ce_loss(val_clip_logits, y_val)
            val_bce = focal_loss_with_logits(val_clip_logits, y_val)
            val_loss = val_ce + (bce_weight * val_bce)

            val_loss_sum += val_loss.item()
            val_ce_sum += val_ce.item()
            val_bce_sum += val_bce.item()

            all_preds.append(torch.sigmoid(val_clip_logits).cpu().numpy())
            all_targets.append(y_val.cpu().numpy())

    num_batches = len(val_loader)
    avg_loss = val_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_ce = val_ce_sum / num_batches if num_batches > 0 else 0.0
    avg_bce = val_bce_sum / num_batches if num_batches > 0 else 0.0
    roc = compute_macro_roc_auc(np.concatenate(all_preds), np.concatenate(all_targets)) if num_batches > 0 else 0.0

    return avg_loss, avg_ce, avg_bce, roc


# =====================================================================
# 3. DATASET
# =====================================================================
class CleanAudioDataset(Dataset):
    def __init__(self, df, class_to_idx, data_dir='data/birdclef-2026', sr=32000, duration=5.0):
        self.df = df
        self.class_to_idx = class_to_idx
        self.audio_dir = os.path.join(data_dir, "train_audio")
        self.sr = sr
        self.target_length = int(duration * sr)

        # Base Spec Only
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

        # Base Log-Mel Extraction ONLY
        spec = self.mel_transform(waveform)
        spec_db = self.db_transform(spec)
        spec_norm = spec_db / 80.0

        # Required Shape:[Time, Freq]
        X = spec_norm.squeeze(0).transpose(0, 1)

        return X, Y


def collate_clean(batch):
    X, Y = zip(*batch)
    return torch.stack(X).transpose(0, 1), torch.stack(Y)


# =====================================================================
# 4. MAIN SCRIPT
# =====================================================================
if __name__ == '__main__':

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'tensorboard/birdclef_train_{timestamp}'
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

    df_full = pd.read_csv(os.path.join(data_dir, "train.csv"))
    label_col = 'primary_label' if 'primary_label' in df_full.columns else df_full.columns[1]
    classes = sorted(df_full[label_col].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    train_df = df_full.sample(frac=0.9, random_state=42)
    val_df = df_full.drop(train_df.index)

    train_ds = CleanAudioDataset(train_df, class_to_idx, data_dir=data_dir, duration=5.0)
    val_ds = CleanAudioDataset(val_df, class_to_idx, data_dir=data_dir, duration=5.0)

    print(f"Training on {len(train_ds)} 5s crops (Train_Audio).")
    print(f"Validating on {len(val_ds)} 5s crops (Train_Audio).")

    batch_size = 50
    grad_accum_steps = 1

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_clean,
                                  num_workers=4)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_clean, num_workers=4)

    # Note: Default architecture params modified inside the imported class to handle 768 In-Features
    model = BirdClassifierSNN(num_classes=num_classes).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters(): param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    snn_params = model.get_param_count()
    print(f"\nTotal Params: {total_params:,} -> SNN: {snn_params:,} | Non-SNN: {total_params - snn_params:,}")
    print(f"Num Train Batches: {len(train_dataloader):,}\n")

    # Instantiate Delta Function on GPU
    delta_fn = torchaudio.transforms.ComputeDeltas().to(device)

    # -------------------------------------------------------------
    # RGB SANITY VISUALIZATION
    # -------------------------------------------------------------
    x_sanity_raw, y_sanity = next(iter(train_dataloader))
    x_sanity_raw = x_sanity_raw.to(device)

    # Extract features WITH NOISE so the sanity check shows exactly what the model sees
    x_feat_sanity = extract_features(x_sanity_raw, delta_fn, training=True)

    print(f"\nSanity Check - X_feat shape (Time, Batch, Freq): {x_feat_sanity.shape} | Y shape: {y_sanity.shape}")

    # Permute to[Batch, Freq, Time]
    x_batch = x_feat_sanity.permute(1, 2, 0)

    # Split into Spec, Delta, and Accel (each is[Batch, 256, Time])
    spec_batch, delta_batch, accel_batch = torch.chunk(x_batch, 3, dim=1)

    components = [
        ("Spectrogram", spec_batch),
        ("Delta", delta_batch),
        ("Acceleration", accel_batch)
    ]

    for name, comp in components:
        c_mean = comp.mean().item()
        c_min = comp.min().item()
        c_max = comp.max().item()

        stat_str = f"{name} - Mean: {c_mean:.4f} | Min: {c_min:.4f} | Max: {c_max:.4f}"
        print(stat_str)

        val = torch.clamp(comp, min=-1.0, max=1.0)
        pos = torch.clamp(val, min=0)
        neg = torch.clamp(-val, min=0)

        r_channel = 1.0 - neg
        g_channel = 1.0 - pos - neg
        b_channel = 1.0 - pos

        rgb_batch = torch.stack([r_channel, g_channel, b_channel], dim=1)
        tt.plot.render_image(rgb_batch, title=stat_str)

    print("Rendered 3 separate batch images (White Bg, Red=Pos, Blue=Neg).\n")
    # -------------------------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    bce_weight = float(np.log(num_classes) / np.log(2.0))
    print(f"BCE Scale Weight calculated as: {bce_weight:.4f}")

    optimizer_steps, accum_steps, num_epochs = 0, 0, 15
    total_steps = num_epochs * max(1, len(train_dataloader)) // grad_accum_steps
    warmup_steps = max(1, total_steps // 10)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    ], milestones=[warmup_steps])

    for e in range(num_epochs):
        model.train()
        train_loss_accum = 0.0
        train_ce_accum = 0.0
        train_bce_accum = 0.0
        train_preds_accum = []
        train_targets_accum = []

        for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"TRAIN - E{e}"):
            x, y = x.to(device), y.to(device)
            x_feat = extract_features(x, delta_fn, training=True)

            model.zero_states()
            all_step_logits = []

            for t in range(x.size(0)):
                logits = model(x_feat[t])
                all_step_logits.append(logits)

            clip_logits = torch.stack(all_step_logits).mean(dim=0)

            ce_loss = custom_multi_class_ce_loss(clip_logits, y)
            bce_loss = focal_loss_with_logits(clip_logits, y)
            loss_unscaled = ce_loss + (bce_weight * bce_loss)

            loss = loss_unscaled / grad_accum_steps
            loss.backward()

            accum_steps += 1
            train_loss_accum += loss_unscaled.item()
            train_ce_accum += ce_loss.item()
            train_bce_accum += bce_loss.item()

            train_preds_accum.append(torch.sigmoid(clip_logits).detach().cpu().numpy())
            train_targets_accum.append(y.cpu().numpy())

            if accum_steps % grad_accum_steps == 0:
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], optimizer_steps)
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1

                update_ema_model(model, ema_model)
                optimizer.zero_grad()

                # Calculate ROC AUC on the recent accumulation batch safely
                batch_roc = compute_macro_roc_auc(np.concatenate(train_preds_accum),
                                                  np.concatenate(train_targets_accum))

                writer.add_scalars("Loss", {"Train": train_loss_accum / grad_accum_steps}, optimizer_steps)
                writer.add_scalars("CE_Loss", {"Train": train_ce_accum / grad_accum_steps}, optimizer_steps)
                writer.add_scalars("BCE_Loss", {"Train": train_bce_accum / grad_accum_steps}, optimizer_steps)
                writer.add_scalars("ROC_AUC", {"Train": batch_roc}, optimizer_steps)

                # Reset batch accumulators
                train_loss_accum, train_ce_accum, train_bce_accum = 0.0, 0.0, 0.0
                train_preds_accum, train_targets_accum = [], []

                if optimizer_steps % 250 == 0:
                    print("\nEvaluating Base Model...")
                    base_loss, base_ce, base_bce, base_roc_auc = evaluate_model(model, val_dataloader, bce_weight,
                                                                                delta_fn)

                    print("Evaluating EMA Model...")
                    ema_loss, ema_ce, ema_bce, ema_roc_auc = evaluate_model(ema_model, val_dataloader, bce_weight,
                                                                            delta_fn)

                    writer.add_scalars("Loss", {"Val_Base": base_loss, "Val_EMA": ema_loss}, optimizer_steps)
                    writer.add_scalars("CE_Loss", {"Val_Base": base_ce, "Val_EMA": ema_ce}, optimizer_steps)
                    writer.add_scalars("BCE_Loss", {"Val_Base": base_bce, "Val_EMA": ema_bce}, optimizer_steps)
                    writer.add_scalars("ROC_AUC", {"Val_Base": base_roc_auc, "Val_EMA": ema_roc_auc}, optimizer_steps)

                    os.makedirs("checkpoints", exist_ok=True)
                    save_file(ema_model.state_dict(),
                              os.path.join("checkpoints", f"model_step_{optimizer_steps}_ema.safetensors"))
                    save_file(model.state_dict(),
                              os.path.join("checkpoints", f"model_step_{optimizer_steps}_base.safetensors"))
                    print(f"\nSaved models. Base ROC-AUC: {base_roc_auc:.4f} | EMA ROC-AUC: {ema_roc_auc:.4f}\n")

                    model.train()

    # Final save after all epochs complete
    os.makedirs("checkpoints", exist_ok=True)
    save_file(ema_model.state_dict(), os.path.join("checkpoints", "model_final_ema.safetensors"))
    save_file(model.state_dict(), os.path.join("checkpoints", "model_final_base.safetensors"))
    print("\nTraining Complete!")
