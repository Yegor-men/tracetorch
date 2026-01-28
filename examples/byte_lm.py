import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tracetorch as tt
from tracetorch import snn
import urllib.request
import math
import matplotlib.pyplot as plt
import os
import urllib.request
from tqdm import tqdm

# ---------------------- basic config ----------------------
torch.manual_seed(0)
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed_all(0)
else:
    device = "cpu"

DATA_DIR = "data"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "tiny_shakespeare.txt")


def download_with_progress(url, path):
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(path):
        print(f"Using cached {path}")
        return
    print(f"Downloading {url} → {path}")
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Download") as t:
        def reporthook(bnum, bsize, tsize):
            t.total = tsize
            t.update(bsize)

        urllib.request.urlretrieve(url, path, reporthook=reporthook)
    print("Done.")


def collate_time_major(batch):
    # batch: list of tuples (x: [L], y: [L]) of length B
    xs = torch.stack([b[0] for b in batch], dim=0)  # [B, L]
    ys = torch.stack([b[1] for b in batch], dim=0)  # [B, L]
    # transpose to [L, B]
    return xs.transpose(0, 1).contiguous(), ys.transpose(0, 1).contiguous()


class ConcatByteSlices(Dataset):
    """
    Dataset of ALL valid length-(seq_len+1) slices from the concatenated byte stream.
    Each __getitem__ returns (x, y) where x,y are 1D tensors of length seq_len.
    Use DataLoader(batch_size=B, collate_fn=collate_time_major) to get [L, B] time-major batches.
    """

    def __init__(self, seq_len=512, split="train", use_bookcorpus=False):
        self.seq_len = int(seq_len)
        self.seq_len_plus1 = self.seq_len + 1

        # load file same as before
        if use_bookcorpus:
            ds = load_dataset("bookcorpus", split="train[:50000]")
            texts = [item['text'] for item in ds]
        else:
            download_with_progress(SHAKESPEARE_URL, SHAKESPEARE_PATH)
            with open(SHAKESPEARE_PATH, "r", encoding="utf-8") as f:
                texts = [f.read()]

        sep = "\n\n---\n\n".encode('utf-8')
        all_bytes = bytearray()
        for text in texts:
            all_bytes.extend(text.encode('utf-8'))
            all_bytes.extend(sep)

        self.data = torch.tensor(list(all_bytes), dtype=torch.long)
        print(f"Total bytes: {len(self.data):,}")

        # all possible start positions where we can take seq_len+1 bytes
        max_start = len(self.data) - self.seq_len_plus1
        if max_start < 0:
            raise ValueError("Not enough data for the requested seq_len")

        starts = torch.arange(0, max_start + 1, dtype=torch.long)

        # train/val split by contiguous ranges of starts (keeps val on later text)
        split_idx = int(len(starts) * 0.9)
        if split == "train":
            self.starts = starts[:split_idx]
        elif split == "val":
            self.starts = starts[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return int(self.starts.numel())

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        s = int(self.starts[idx])
        seg = self.data[s: s + self.seq_len_plus1]  # shape [L+1]
        x = seg[:-1].clone()  # length L
        y = seg[1:].clone()  # length L
        return x, y


seq_len = 256
batch_size = 64

use_bookcorpus = False
train_dataset = ConcatByteSlices(seq_len=seq_len, split="train", use_bookcorpus=use_bookcorpus)
val_dataset = ConcatByteSlices(seq_len=seq_len, split="val", use_bookcorpus=use_bookcorpus)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_time_major,
    drop_last=True,
    num_workers=0  # <- no multiprocessing, simplest and reliable
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_time_major,
    drop_last=False,
    num_workers=0  # <- keep sync single-process loading
)


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
                "To be, or not to be",
                "Thrice the branded cat hath mewed",
                "Shall I compare thee to a summer's day?"
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
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.lin.weight, 0.0, 0.01)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return x + self.lin(self.lif(x))


class SNNLM(snn.TTModule):
    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            num_layers: int,
            emb_dropout: float = 0.05,
            dec_dropout: float = 0.05,
    ):
        super().__init__()

        self.emb = nn.Sequential(
            nn.Embedding(256, embed_dim),
            nn.Dropout(emb_dropout),
            nn.Linear(embed_dim, hidden_dim)
        )
        nn.init.xavier_uniform_(self.emb[-1].weight)
        nn.init.zeros_(self.emb[-1].bias)

        layers = [ResidualSpike(hidden_dim) for _ in range(num_layers)]
        self.net = nn.Sequential(*layers)

        self.dec = nn.Sequential(
            # snn.SRReadout(
            #     hidden_dim,
            #     alpha=torch.rand(hidden_dim) /2,
            #     beta=torch.rand(hidden_dim) / 2,
            #     gamma=torch.rand(hidden_dim) / 2,
            # ),
            # nn.Dropout(dec_dropout),
            nn.Linear(hidden_dim, 256)
        )
        nn.init.xavier_uniform_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        emb_byte = self.emb(x)
        att_byte = self.net(emb_byte)
        pred_byte = self.dec(att_byte)

        return pred_byte


model = SNNLM(256, 256, 10).to(device)
print(f"\nNum params: {model.get_param_count():,}")
print(f"num batches: {len(train_dataloader):,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 1


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


train_losses = []
optimizer_steps = 0

for e in range(num_epochs):
    # TRAIN
    model.train()
    model.zero_states()
    train_loss = 0
    for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        x = x.to(device)  # [B, L]
        y = y.to(device)
        x = add_byte_noise(x)

        model.detach_states()
        model.zero_states()
        running_loss = 0.0

        for t in range(x.size(0)):
            x_t, y_t = x[t], y[t]
            model_output = model(x_t)
            loss = loss_fn(model_output, y_t)
            running_loss = running_loss + loss

        running_loss = running_loss / x.size(0)
        train_losses.append(running_loss.item())
        train_loss += train_losses[-1]

        model.zero_grad()
        running_loss.backward()
        optimizer.step()
        optimizer_steps += 1

        if (optimizer_steps) % 10 == 0 and optimizer_steps != 0:
            avg_loss = sum(train_losses[-10:]) / 10
            bits_per_byte = avg_loss / math.log(2)
            perplex = math.exp(avg_loss)
            plt.title(
                f"STEP: {optimizer_steps:,} | LOSS: {avg_loss:.5f} | BPB: {bits_per_byte:5f} | PERPLEXITY: {perplex:.1f}")
            plt.plot(train_losses)
            plt.show()

        if (optimizer_steps + 1) % 100 == 0:
            # sample config example: temperature + top_k
            sample_cfg = {"temperature": 1.0, "top_k": 10, "top_p": 0.95}
            out_fn = os.path.join("data", f"generated_epoch_{e}.txt")
            save_generation_file(model, out_fn, gen_length=100, sample_config=sample_cfg)
            print(f"Saved generation to {out_fn}")

    train_loss /= len(train_dataloader)

    # ---------------------- TEST ----------------------
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(val_dataloader, total=len(val_dataloader), desc=f"TEST - E{e}"):
            x = x.to(device)  # [B, L]
            y = y.to(device)

            model.zero_states()
            running_loss = 0.0

            for t in range(x.size(0)):
                x_t, y_t = x[t], y[t]
                model_output = model(x_t)
                loss = loss_fn(model_output, y_t)
                running_loss = running_loss + loss

            running_loss = running_loss / x.size(0)
            test_loss += running_loss.item()
    test_loss /= len(val_dataloader)

    print(f"TRAIN: {train_loss:.5f} | TEST: {test_loss:.5f}")
