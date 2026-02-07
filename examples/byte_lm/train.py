import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tracetorch as tt
from tracetorch import snn
import math
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import bisect
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from safetensors.torch import save_file, load_file

use_amp = True
amp_dtype = torch.bfloat16

# ---------------------- basic config ----------------------
torch.manual_seed(0)
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed_all(0)
else:
    device = "cpu"


class WikiTextDataset(Dataset):
    def __init__(self, data_dir='data', split='train', seq_len=512):
        self.seq_len = seq_len
        self.split = split
        config_name = 'wikitext-103-raw-v1'
        ds = load_dataset('Salesforce/wikitext', config_name, cache_dir=data_dir)
        hf_split = 'train' if split == 'train' else 'validation'
        self.dataset = ds[hf_split]

        # Pre-compute all valid (row_idx, start_offset) tuples
        self.samples = []
        for i in tqdm(range(len(self.dataset)), desc=f"Indexing {split} chunks"):
            text = self.dataset[i]['text']

            # Efficient length check. 
            # Note: We use seq_len + 1 because we need context + target (shifted by 1)
            byte_len = len(text.encode('utf-8'))

            # We want patches of length seq_len + 1.
            # Stride by seq_len so there is no overlap between X inputs of consecutive samples
            if byte_len >= self.seq_len + 1:
                # Range max is inclusive of data, so exclusive limit is byte_len - seq_len
                for start in range(0, byte_len - self.seq_len, self.seq_len):
                    self.samples.append((i, start))

        print(f"Loaded {split} split: {len(self.samples)} samples (chunks).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx, start = self.samples[idx]
        text = self.dataset[row_idx]['text']
        byte_text = text.encode('utf-8')

        # Slice the exact window of seq_len + 1 bytes
        end = start + self.seq_len + 1
        slice_bytes = byte_text[start:end]

        tensor = torch.tensor(list(slice_bytes), dtype=torch.long)
        return tensor[:-1], tensor[1:]  # X: [seq_len], Y: [seq_len] shifted


# Usage example
def get_dataloaders(batch_size=64, seq_len=512, num_workers=0, data_dir='data'):
    train_ds = WikiTextDataset(data_dir=data_dir, split='train', seq_len=seq_len)
    test_ds = WikiTextDataset(data_dir=data_dir, split='test', seq_len=seq_len)

    # Train: Shuffle=True ensures random order of batches every epoch
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)

    # Test: Sequential is fine
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, test_loader


# Custom collate to stack into [L, B]
def collate_fn(batch):
    X, Y = zip(*batch)
    X = torch.stack(X).T  # [L, B]
    Y = torch.stack(Y).T  # [L, B]
    return X, Y


# ---------------------- build data & model ----------------------


def add_byte_noise(tensor: torch.Tensor, p: float = 0, device=None) -> torch.Tensor:
    if p <= 0:
        return tensor
    if device is None:
        device = tensor.device

    # Create mask of shape [L, B]
    mask = torch.rand(tensor.shape, device=device) < p

    # Generate random replacements only where needed
    noise = torch.randint(0, 256, size=tensor.shape, device=device, dtype=tensor.dtype)

    # Apply: keep original where mask=False, replace where True
    noised = torch.where(mask, noise, tensor)

    return noised


if __name__ == '__main__':
    import subprocess
    import webbrowser
    import time
    import atexit
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'tensorboard/byte_lm_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)
    subprocess.Popen(['tensorboard', '--logdir', "tensorboard", '--port', '6006'])
    time.sleep(2)  # Give TensorBoard time to start
    webbrowser.open('http://localhost:6006')


    def cleanup():
        writer.flush()
        writer.close()


    atexit.register(cleanup)

    batch_size = 16
    minibatch_size = 1
    seq_len = 1024
    train_dataloader, val_dataloader = get_dataloaders(batch_size, seq_len, num_workers=0)

    from architecture import SNNLM

    model = SNNLM(1024, 10).to(device)
    # model.load_state_dict(load_file("checkpoints/step_20300_e2_bpb15334.safetensors"))
    print(f"\nNum params: {model.get_param_count():,}")
    print(f"num batches: {len(train_dataloader):,}")

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False


    @torch.no_grad()
    def update_ema_model(model, ema_model, decay: float = 0.999):
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_steps = 0
    accum_steps = 0
    num_epochs = 1

    total_steps = num_epochs * len(train_dataloader) // minibatch_size
    warmup_steps = 1000
    cosine_steps = total_steps - warmup_steps
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  # start at 1% of base_lr
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,  # full cosine over remaining steps
        eta_min=1e-6  # or 1e-5, very small floor
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    for e in range(num_epochs):

        # TRAIN
        model.train()
        ema_model.eval()
        train_loss = 0.0
        for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"TRAIN - E{e}"):
            x = x.to(device)  # [L, B]
            y = y.to(device)
            x = add_byte_noise(x)

            model.detach_states()
            model.zero_states()
            running_loss = 0.0

            for t in range(x.size(0)):
                x_t, y_t = x[t], y[t]
                model_output = model(x_t)
                loss_t = loss_fn(model_output, y_t)
                running_loss += loss_t

            running_loss = running_loss / x.size(0)
            scaled_loss = running_loss / minibatch_size

            scaled_loss.backward()
            accum_steps += 1

            train_loss += running_loss.item()

            if accum_steps % minibatch_size == 0 and accum_steps != 0:
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("learning_rate", current_lr, optimizer_steps)

                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                update_ema_model(model, ema_model)
                optimizer.zero_grad()

                train_loss /= minibatch_size
                train_bpb = train_loss / math.log(2)
                train_ppl = math.exp(train_loss)

                writer.add_scalars("loss", {"base": train_loss}, optimizer_steps)
                writer.add_scalars("bpb", {"base": train_bpb}, optimizer_steps)
                writer.add_scalars("ppl", {"base": train_ppl}, optimizer_steps)

                train_loss = 0.0

            if optimizer_steps % 100 == 0 and optimizer_steps != 0:
                val_loss = 0.0
                ema_model.eval()
                with torch.no_grad():
                    for j, (x, y) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="VAL"):
                        x = x.to(device)  # [L, B]
                        y = y.to(device)

                        ema_model.zero_states()
                        running_loss = 0.0

                        for t in range(x.size(0)):
                            x_t, y_t = x[t], y[t]
                            model_output = ema_model(x_t)
                            loss_t = loss_fn(model_output, y_t)
                            running_loss += loss_t

                        running_loss = running_loss / x.size(0)
                        val_loss += running_loss.item()

                val_loss /= len(val_dataloader)
                val_bpb = val_loss / math.log(2)
                val_ppl = math.exp(val_loss)

                writer.add_scalars("loss", {"ema": val_loss}, optimizer_steps)
                writer.add_scalars("bpb", {"ema": val_bpb}, optimizer_steps)
                writer.add_scalars("ppl", {"ema": val_ppl}, optimizer_steps)

                from inference import sample

                sample_cfg = {"temperature": 0.5, "top_k": 20, "top_p": 0.9}
                sample_filename = os.path.join("samples", f"generated_step_{optimizer_steps}.txt")
                sample(
                    model=ema_model,
                    device=device,
                    out_path=sample_filename,
                    sample_cfg=sample_cfg,
                    gen_length=1000
                )
                print(f"Saved generation to {sample_filename}")

                os.makedirs("checkpoints", exist_ok=True)
                model_path = os.path.join("checkpoints", f"model_step_{optimizer_steps}.safetensors")
                save_file(ema_model.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

    writer.close()
