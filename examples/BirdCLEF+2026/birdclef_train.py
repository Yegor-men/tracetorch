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
from safetensors.torch import save_file, load_file
import subprocess
import webbrowser
import time
import atexit
from sklearn.metrics import roc_auc_score

import tracetorch as tt
from tracetorch import snn

# Fix the 'too many fds' multiprocessing error
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.manual_seed_all(0)


# =====================================================================
# 1. ARCHITECTURE
# =====================================================================
def foobar(x):
    return nn.functional.sigmoid(4.0 * x)


dsrlits_min_timescale = tt.functional.halflife_to_decay(1)
dsrlits_max_timescale = tt.functional.halflife_to_decay(50)
dsrlits_diff = dsrlits_max_timescale - dsrlits_min_timescale


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


class BirdClassifierSNN(snn.TTModel):
    def __init__(self, in_features=256, hidden_features=1024, num_layers=10, num_classes=234):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_features, hidden_features), nn.Dropout(0.0))
        self.net = nn.Sequential(*[ResidualSpike(hidden_features) for _ in range(num_layers)])
        self.dec = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_features, num_classes)
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        return self.dec(self.net(self.enc(x)))


# =====================================================================
# 2. UTILS & CUSTOM LOSS
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


def add_spec_noise(tensor: torch.Tensor, p: float = 0.1, device=None) -> torch.Tensor:
    if p <= 0: return tensor
    if device is None: device = tensor.device
    mask = torch.rand(tensor.shape, device=device) > p
    return tensor * mask


def custom_multi_class_ce_loss(logits, targets):
    """
    Computes avg(-ln(prob)) ONLY for the positive target classes.
    Softmax forces the negatives toward zero implicitly.
    """
    lsm = torch.nn.functional.log_softmax(logits, dim=1)
    sum_lsm_positives = (targets * lsm).sum(dim=1)
    num_positives = torch.clamp(targets.sum(dim=1), min=1.0)
    loss_per_sample = -sum_lsm_positives / num_positives
    return loss_per_sample.mean()


def focal_loss_with_logits(logits, targets, gamma=2.0):
    """
    Focal Loss pushes the model to ignore 'easy' predictions (like the 233 background classes)
    and focus all gradient energy on hard mistakes (false positives/negatives).
    """
    # Get raw BCE loss (unreduced so we can scale each class independently)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # Calculate probabilities
    probs = torch.sigmoid(logits)

    # p_t is the probability of the TRUE class
    # If target=1, p_t = probs. If target=0, p_t = 1 - probs.
    p_t = probs * targets + (1 - probs) * (1 - targets)

    # Modulating factor: (1 - p_t)^gamma
    # If model is 99% sure and correct, weight becomes (1 - 0.99)^2 = 0.0001 (gradient vanishes)
    focal_weight = (1 - p_t) ** gamma

    # Apply weight and return mean
    return (focal_weight * bce_loss).mean()


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

        spec = self.mel_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0)
        X = (spec_db / 80.0).transpose(0, 1)
        return X, Y


def collate_clean(batch):
    X, Y = zip(*batch)
    return torch.stack(X).transpose(0, 1), torch.stack(Y)


def find_free_port(start_port=6006):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0: return port
        port += 1


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

    model = BirdClassifierSNN(num_classes=num_classes).to(device)

    # Uncomment to load previous state
    # model.load_state_dict(load_file("checkpoints/_model_step_1750_base.safetensors"))

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters(): param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    snn_params = model.get_param_count()
    print(f"\nTotal Params: {total_params:,} -> SNN: {snn_params:,} | Non-SNN: {total_params - snn_params:,}")
    print(f"Num Train Batches: {len(train_dataloader):,}\n")

    x_sanity, y_sanity = next(iter(train_dataloader))
    print(f"Sanity Check - X shape (Time, Batch, Freq): {x_sanity.shape} | Y shape: {y_sanity.shape}")
    print(f"Sanity Check - X Mean: {x_sanity.mean():.4f} | Min: {x_sanity.min():.4f} | Max: {x_sanity.max():.4f}")

    x_vis = x_sanity.permute(1, 2, 0).unsqueeze(1)
    tt.plot.render_image(x_vis)
    print("Rendered Sanity Check Image.\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # -----------------------------------------------------------------
    # BALANCED DUAL-LOSS SETUP
    # -----------------------------------------------------------------
    # Initial Custom CE is ~ ln(num_classes). Initial BCE is ~ ln(2).
    # Scale BCE up so both losses exert equal force at initialization.
    # For 234 classes, bce_weight will be exactly ~7.87.
    # bce_loss_fn = nn.BCEWithLogitsLoss()
    bce_weight = float(np.log(num_classes) / np.log(2.0))
    print(f"BCE Scale Weight calculated as: {bce_weight:.4f}")
    # -----------------------------------------------------------------

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
            x = add_spec_noise(x, p=0.1)

            model.zero_states()
            all_step_logits = []

            for t in range(x.size(0)):
                logits = model(x[t])
                all_step_logits.append(logits)

            all_step_logits = torch.stack(all_step_logits)
            clip_logits = torch.mean(all_step_logits, dim=0)

            # Compute combined balanced loss
            ce_loss = custom_multi_class_ce_loss(clip_logits, y)
            # bce_loss = bce_loss_fn(clip_logits, y)
            bce_loss = focal_loss_with_logits(clip_logits, y)

            loss = ce_loss + (bce_weight * bce_loss)
            (loss / grad_accum_steps).backward()

            accum_steps += 1
            train_loss_accum += loss.item()
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

                batch_roc = compute_macro_roc_auc(np.concatenate(train_preds_accum),
                                                  np.concatenate(train_targets_accum))

                # Log all 3 separately to TensorBoard
                writer.add_scalars("Loss", {"Train": train_loss_accum / grad_accum_steps}, optimizer_steps)
                writer.add_scalars("CE_Loss", {"Train": train_ce_accum / grad_accum_steps}, optimizer_steps)
                writer.add_scalars("BCE_Loss", {"Train": train_bce_accum / grad_accum_steps}, optimizer_steps)
                writer.add_scalars("ROC_AUC", {"Train": batch_roc}, optimizer_steps)

                train_loss_accum = 0.0
                train_ce_accum = 0.0
                train_bce_accum = 0.0
                train_preds_accum = []
                train_targets_accum = []

                if optimizer_steps % 250 == 0:
                    def evaluate_model(eval_model, name):
                        eval_model.eval()
                        all_preds, all_targets = [], []
                        val_loss_sum = 0.0
                        val_ce_sum = 0.0
                        val_bce_sum = 0.0

                        with torch.no_grad():
                            for x_val, y_val in tqdm(val_dataloader, desc=f"VAL ({name})", leave=False):
                                x_val, y_val = x_val.to(device), y_val.to(device)
                                eval_model.zero_states()

                                val_step_logits = []
                                for t in range(x_val.size(0)):
                                    val_step_logits.append(eval_model(x_val[t]))

                                val_step_logits = torch.stack(val_step_logits)
                                val_clip_logits = torch.mean(val_step_logits, dim=0)

                                val_ce = custom_multi_class_ce_loss(val_clip_logits, y_val)
                                # val_bce = bce_loss_fn(val_clip_logits, y_val)
                                val_bce = focal_loss_with_logits(val_clip_logits, y_val)
                                val_loss = val_ce + (bce_weight * val_bce)

                                val_loss_sum += val_loss.item()
                                val_ce_sum += val_ce.item()
                                val_bce_sum += val_bce.item()

                                probs = torch.sigmoid(val_clip_logits)
                                all_preds.append(probs.cpu().numpy())
                                all_targets.append(y_val.cpu().numpy())

                        num_batches = len(val_dataloader)
                        avg_loss = val_loss_sum / num_batches if num_batches > 0 else 0.0
                        avg_ce = val_ce_sum / num_batches if num_batches > 0 else 0.0
                        avg_bce = val_bce_sum / num_batches if num_batches > 0 else 0.0

                        roc = compute_macro_roc_auc(np.concatenate(all_preds),
                                                    np.concatenate(all_targets)) if num_batches > 0 else 0.0
                        return avg_loss, avg_ce, avg_bce, roc


                    print("\nEvaluating Base Model...")
                    base_loss, base_ce, base_bce, base_roc_auc = evaluate_model(model, "Base")
                    print("Evaluating EMA Model...")
                    ema_loss, ema_ce, ema_bce, ema_roc_auc = evaluate_model(ema_model, "EMA")

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

    save_file(ema_model.state_dict(), os.path.join("checkpoints", "model_final_ema.safetensors"))
    save_file(model.state_dict(), os.path.join("checkpoints", "model_final_base.safetensors"))
