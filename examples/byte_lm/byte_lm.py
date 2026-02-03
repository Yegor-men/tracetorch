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


# ---------------------- generation / sampling helpers (defined once) ----------------------
def encode_string(s):
    return list(s.encode('utf-8'))


def sample_next_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """
    logits: [B, V]
    returns: next indices [B]
    """
    if temperature != 1.0:
        logits = logits / float(temperature)

    logits = logits.clone()

    # top-k: zero out everything not in top_k by setting to -inf
    if top_k is not None and top_k > 0:
        top_k_val = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, top_k_val, dim=-1)  # [B, top_k_val]
        min_vals = vals[..., -1].unsqueeze(-1)  # [B, 1]
        logits = torch.where(logits < min_vals, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype), logits)

    # top-p (nucleus)
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # for each batch row, ban tokens where cumulative prob > top_p (but keep at least one)
        B = logits.size(0)
        V = logits.size(1)
        mask = torch.zeros_like(sorted_probs, dtype=torch.bool)  # [B, V]
        for i in range(B):
            # find positions where cum_probs[i] > top_p
            over = cum_probs[i] > top_p
            if over.any():
                # ban all where over is True (keep earlier ones)
                mask[i, over] = True
                # ensure at least one token remains (never ban the first)
                mask[i, 0] = False
        # set masked logits to -inf in original logits by mapping sorted_idx positions
        for i in range(B):
            ban_positions = sorted_idx[i][mask[i]]
            if ban_positions.numel() > 0:
                logits[i, ban_positions] = -1e9

    probs = F.softmax(logits, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_idx


def generate_greedy_batch(model, start_batch, length):
    """Return tensor shape [T+1, B] where T=length, each column is a generated index sequence (greedy)."""
    device_local = next(model.parameters()).device
    seq = [start_batch.to(device_local)]
    model.detach_states()
    model.zero_states()
    with torch.no_grad():
        for _ in range(length):
            logits = model(seq[-1])  # [B, V]
            next_idx = torch.argmax(logits, dim=-1)  # [B]
            seq.append(next_idx)
    return torch.stack(seq, dim=0)  # [T+1, B]


def generate_sampled_batch(model, start_batch, length, temperature=1.0, top_k=None, top_p=None):
    """Sampled generation with temperature/top-k/top-p. Returns [T+1, B]."""
    device_local = next(model.parameters()).device
    seq = [start_batch.to(device_local)]
    model.detach_states()
    model.zero_states()
    with torch.no_grad():
        for _ in range(length):
            logits = model(seq[-1])  # [B, V]
            next_idx = sample_next_from_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            seq.append(next_idx)
    return torch.stack(seq, dim=0)


def decode_sequence(byte_seq):
    return bytes(byte_seq).decode('utf-8', errors='replace')


def save_generation_file(model, out_path, gen_length=200, sample_config=None, example_seeds=None):
    model.eval()

    start_bytes = torch.arange(0, 256, dtype=torch.long)

    greedy_gen = generate_greedy_batch(model, start_bytes, length=gen_length)

    sampled_gen = None
    if sample_config is not None:
        temperature = sample_config.get("temperature", 1.0)
        top_k = sample_config.get("top_k", None)
        top_p = sample_config.get("top_p", None)
        sampled_gen = generate_sampled_batch(model, start_bytes, length=gen_length,
                                             temperature=temperature, top_k=top_k, top_p=top_p)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== GREEDY generation from byte seeds (0-255) ===\n")
        Tplus1, B = greedy_gen.shape
        for b in range(B):
            seq_bytes = greedy_gen[:, b].tolist()
            seed_byte = seq_bytes[0]
            text = decode_sequence(seq_bytes)
            f.write(f"seed_byte={seed_byte} ({hex(seed_byte)}) → {repr(text[:100])}...\n")

        if sampled_gen is not None:
            f.write("\n=== SAMPLED generation from byte seeds ===\n")
            for b in range(B):
                seq_bytes = sampled_gen[:, b].tolist()
                seed_byte = seq_bytes[0]
                text = decode_sequence(seq_bytes)
                f.write(f"seed_byte={seed_byte} ({hex(seed_byte)}) → {repr(text[:100])}...\n")

        if example_seeds is None:
            example_seeds = [
                "Skibidi Toilet (stylized as skibidi toilet) is an animated web series created by ",
                "Walter Hartwell White, also known by his alias Heisenberg",
                'Gustavo "Gus" Fring (Spanish pronunciation: [gusˈtaβo ˈfɾin]) is a fictional character portrayed by Giancarlo Esposito in the Breaking Bad crime drama franchise.'
            ]

        f.write("\n=== CUSTOM SEED EXAMPLES ===\n")
        for seed_str in example_seeds:
            seed_bytes = encode_string(seed_str)

            # Greedy
            model.detach_states()
            model.zero_states()
            cur = torch.tensor([seed_bytes[0]], dtype=torch.long, device=device)
            generated = [cur.item()]
            with torch.no_grad():
                for b in seed_bytes[1:]:
                    _ = model(cur)
                    generated.append(b)
                    cur = torch.tensor([b], dtype=torch.long, device=device)
                for _ in range(gen_length):
                    logits = model(cur)
                    cur = torch.argmax(logits, dim=-1)
                    generated.append(cur.item())
            s_greedy = decode_sequence(generated)

            # Sampled (if enabled)
            s_sampled = "(sampling disabled)"
            if sample_config is not None:
                model.detach_states()
                model.zero_states()
                cur = torch.tensor([seed_bytes[0]], dtype=torch.long, device=device)
                generated_s = [cur.item()]
                with torch.no_grad():
                    for b in seed_bytes[1:]:
                        _ = model(cur)
                        generated_s.append(b)
                        cur = torch.tensor([b], dtype=torch.long, device=device)
                    for _ in range(gen_length):
                        logits = model(cur)
                        cur = sample_next_from_logits(logits, **sample_config)
                        generated_s.append(cur.item())
                s_sampled = decode_sequence(generated_s)

            f.write(f"SEED: {repr(seed_str)}\n")
            f.write(f"  GREEDY  → {s_greedy}\n")
            f.write(f"  SAMPLED → {s_sampled}\n\n")

    print(f"Wrote generation output to {out_path}")


# ---------------------- build data & model ----------------------

class ResidualSpike(snn.TTModule):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lif = snn.BSRLIF(
            hidden_dim,
            alpha=torch.rand(hidden_dim),
            beta=torch.rand(hidden_dim),
            gamma=torch.rand(hidden_dim),
            pos_threshold=torch.rand(hidden_dim),
            neg_threshold=torch.rand(hidden_dim),
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.lin.weight, 0.0, 0.01)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        spk = self.lif(x)
        delta = self.lin(spk)
        return x + delta


class SNNLM(snn.TTModule):
    def __init__(
            self,
            hidden_dim: int,
            num_layers: int,
            emb_dropout: float = 0.1,
            dec_dropout: float = 0.1,
    ):
        super().__init__()

        self.emb = nn.Sequential(
            nn.Embedding(256, hidden_dim),
            nn.Dropout(emb_dropout),
        )

        layers = [ResidualSpike(hidden_dim) for _ in range(num_layers)]
        self.net = nn.Sequential(*layers)

        self.dec = nn.Sequential(
            nn.Dropout(dec_dropout),
            nn.Linear(hidden_dim, 256)
        )
        # nn.init.xavier_uniform_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        emb_byte = self.emb(x)
        att_byte = self.net(emb_byte)
        pred_byte = self.dec(att_byte)

        return pred_byte


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
    batch_size = 32
    seq_len = 512
    train_dataloader, val_dataloader = get_dataloaders(batch_size, seq_len, num_workers=4)  # Try 4, if issues set 0

    model = SNNLM(2048, 10).to(device)
    # model.load_state_dict(torch.load("data/baseline_model_5200_step.pt"))
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
    timestep_losses = [-math.log(1 / 256) for _ in range(seq_len)]
    ema_timestep_losses = [-math.log(1 / 256) for _ in range(seq_len)]

    total_steps = 1000  # Or len(train_dataloader) if you want to use it
    train_losses = []
    ema_train_losses = []
    val_losses = []

    num_epochs = 10
    for e in range(num_epochs):

        # TRAIN
        model.train()
        ema_model.eval()
        for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"TRAIN - E{e}"):
            x = x.to(device)  # [L, B]
            y = y.to(device)
            x = add_byte_noise(x)

            model.detach_states()
            model.zero_states()
            running_loss = 0.0
            ema_running_loss = 0.0

            for t in range(x.size(0)):
                x_t, y_t = x[t], y[t]
                model_output = model(x_t)
                loss_t = loss_fn(model_output, y_t)
                running_loss += loss_t
                timestep_losses[t] *= 0.95
                timestep_losses[t] += loss_t.item() * 0.05

                with torch.no_grad():
                    ema_model_output = ema_model(x_t)
                    ema_loss_t = loss_fn(ema_model_output, y_t)
                    ema_running_loss += ema_loss_t
                    ema_timestep_losses[t] *= 0.95
                    ema_timestep_losses[t] += ema_loss_t.item() * 0.05

            running_loss = running_loss / x.size(0)
            with torch.no_grad():
                ema_running_loss = ema_running_loss / x.size(0)

            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()
            optimizer_steps += 1
            update_ema_model(model, ema_model)

            train_losses.append(running_loss.item())
            ema_train_losses.append(ema_running_loss.item())

            if optimizer_steps % 10 == 0 and optimizer_steps != 0:
                avg_loss = sum(train_losses[-10:]) / 10
                bits_per_byte = avg_loss / math.log(2)
                perplex = math.exp(avg_loss)
                plt.title(
                    f"STEP: {optimizer_steps:,} | LOSS: {avg_loss:.5f} | BPB: {bits_per_byte:.5f} | PPL: {perplex:.1f}")
                plt.plot(train_losses, label="base")
                plt.plot(ema_train_losses, label="ema")
                plt.legend()
                plt.show()

                plt.title(f"Timestep losses")
                plt.plot(timestep_losses, label="base")
                plt.plot(ema_timestep_losses, label="ema")
                plt.legend()
                plt.show()

            if optimizer_steps % 100 == 0:
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
                val_losses.append(val_loss)

                plt.title(f"Validation loss")
                plt.plot(val_losses, label="val")
                plt.legend()
                plt.show()

                sample_cfg = {"temperature": 1.0, "top_k": 10, "top_p": 0.95}
                out_fn = os.path.join("samples", f"generated_step_{optimizer_steps}.txt")
                save_generation_file(ema_model, out_fn, gen_length=500, sample_config=sample_cfg)
                print(f"Saved generation to {out_fn}")

                model_path = os.path.join("checkpoints", f"model_step_{optimizer_steps}.pt")
                torch.save(ema_model.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

        avg_train_loss = sum(train_losses) / len(train_losses)
