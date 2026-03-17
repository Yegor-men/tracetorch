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
import librosa
import subprocess
import webbrowser
import time
import atexit

from birdclef_architecture import SNNWorldModel, PredictiveDecoder

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


class UnlabeledChunkDataset(Dataset):
    def __init__(self, filenames, data_dir='data/birdclef-2026', sr=32000, duration=5.0):
        self.audio_dir = os.path.join(data_dir, "train_soundscapes")
        self.sr = sr
        self.duration = duration
        self.num_samples = int(duration * sr)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=2048, hop_length=512, n_mels=256, power=2.0)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # Expand every 60s file into 12 discrete 5s chunks (0 to 11)
        self.samples = [(f, idx) for f in filenames for idx in range(12)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, chunk_idx = self.samples[idx]
        file_path = os.path.join(self.audio_dir, filename)

        offset = chunk_idx * self.duration

        try:
            waveform_np, _ = librosa.load(file_path, sr=self.sr, mono=True, offset=offset, duration=self.duration)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        except Exception:
            waveform = torch.zeros(1, self.num_samples)

        if waveform.shape[1] < self.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))

        spec = self.mel_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0)
        X = ((spec_db + 80.0) / 80.0).transpose(0, 1)  # [~313, 256]
        return X


def collate_unlabeled(batch):
    X = torch.stack(batch).transpose(0, 1)  # [Time, Batch, 256]
    return X


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'tensorboard/birdclef_pretrain_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)

    subprocess.Popen(['tensorboard', '--logdir', "tensorboard", '--port', '6006'])
    time.sleep(2)
    webbrowser.open('http://localhost:6006')


    def cleanup():
        writer.flush()
        writer.close()


    atexit.register(cleanup)

    data_dir = 'data/birdclef-2026'

    # Filter out labeled files to avoid contamination
    ss_csv_path = os.path.join(data_dir, "train_soundscapes_labels.csv")
    df_ss_all = pd.read_csv(ss_csv_path)
    labeled_files = set([f + '.ogg' if not f.endswith('.ogg') else f for f in df_ss_all['filename'].unique()])

    all_files = set(os.listdir(os.path.join(data_dir, "train_soundscapes")))
    unlabeled_files = [f for f in all_files if f.endswith('.ogg') and f not in labeled_files]
    random.shuffle(unlabeled_files)

    split_idx = int(len(unlabeled_files) * 0.9)  # 90/10 split
    train_ds = UnlabeledChunkDataset(unlabeled_files[:split_idx], data_dir=data_dir)
    val_ds = UnlabeledChunkDataset(unlabeled_files[split_idx:], data_dir=data_dir)

    print(f"Pretraining on {len(train_ds)} chunks (5s each).")
    print(f"Validating on {len(val_ds)} chunks (5s each).")

    # Massively scaled up batch size thanks to much shorter sequence lengths
    batch_size = 50
    grad_accum_steps = 1

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_unlabeled,
                                  num_workers=4)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_unlabeled,
                                num_workers=4)

    world_model = SNNWorldModel().to(device)
    pred_decoder = PredictiveDecoder().to(device)

    ema_world_model, ema_pred_decoder = copy.deepcopy(world_model), copy.deepcopy(pred_decoder)
    ema_world_model.eval()
    ema_pred_decoder.eval()
    for p in ema_world_model.parameters(): p.requires_grad = False
    for p in ema_pred_decoder.parameters(): p.requires_grad = False

    optimizer = torch.optim.AdamW(list(world_model.parameters()) + list(pred_decoder.parameters()), 1e-4)
    loss_fn = nn.MSELoss()

    optimizer_steps, accum_steps, num_epochs = 0, 0, 2
    total_steps = num_epochs * max(1, len(train_dataloader)) // grad_accum_steps
    warmup_steps = max(1, total_steps // 10)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    ], milestones=[warmup_steps])

    for e in range(num_epochs):
        world_model.train()
        pred_decoder.train()
        train_loss_accum = 0.0

        for i, x in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"PRETRAIN - E{e}"):
            x = x.to(device)
            x = add_spec_noise(x, p=0.1)

            x_in, x_tgt = x[:-1], x[1:]

            world_model.detach_states()
            world_model.zero_states()

            step_losses = 0.0

            for t in range(x_in.size(0)):
                latents = world_model(x_in[t])
                pred = pred_decoder(latents)
                step_losses += loss_fn(pred, x_tgt[t])

            loss = step_losses / x_in.size(0)
            (loss / grad_accum_steps).backward()

            accum_steps += 1
            train_loss_accum += loss.item()

            if accum_steps % grad_accum_steps == 0:
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], optimizer_steps)
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1

                update_ema_model(world_model, ema_world_model)
                update_ema_model(pred_decoder, ema_pred_decoder)
                optimizer.zero_grad()

                writer.add_scalars("Loss", {"Train_MSE": train_loss_accum / grad_accum_steps}, optimizer_steps)
                train_loss_accum = 0.0

            if optimizer_steps % 500 == 0 and optimizer_steps != 0:
                ema_world_model.eval()
                ema_pred_decoder.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for x_val in tqdm(val_dataloader, desc="VAL"):
                        x_val = x_val.to(device)
                        x_in_val, x_tgt_val = x_val[:-1], x_val[1:]

                        ema_world_model.zero_states()
                        step_losses_val = 0.0

                        for t in range(x_in_val.size(0)):
                            step_losses_val += loss_fn(ema_pred_decoder(ema_world_model(x_in_val[t])), x_tgt_val[t])

                        val_loss += (step_losses_val / x_in_val.size(0)).item()

                if len(val_dataloader) > 0:
                    writer.add_scalars("Loss", {"Val_MSE": val_loss / len(val_dataloader)}, optimizer_steps)

                os.makedirs("checkpoints", exist_ok=True)
                save_file(ema_world_model.state_dict(),
                          os.path.join("checkpoints", f"world_model_step_{optimizer_steps}_ema.safetensors"))
                save_file(ema_pred_decoder.state_dict(),
                          os.path.join("checkpoints", f"pred_decoder_step_{optimizer_steps}_ema.safetensors"))
                world_model.train()
                pred_decoder.train()

    os.makedirs("checkpoints", exist_ok=True)
    save_file(ema_world_model.state_dict(), os.path.join("checkpoints", "world_model_final_ema.safetensors"))
