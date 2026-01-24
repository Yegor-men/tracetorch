#!/usr/bin/env python3
import os
import torch
from torch import nn
import torch.nn.functional as F
import tracetorch as tt
from tracetorch import snn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import urllib.request
import math

# ---------------------- basic config ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "tiny_shakespeare.txt")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# ---------------------- data download / vocab / dataset ----------------------
def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(URL, DATA_PATH)
        print("Done.")


def build_vocab(text):
    """Return (char_to_idx, idx_to_char) built from the provided text."""
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_idx, idx_to_char


class CharDataset(Dataset):
    def __init__(self, text, seq_len, char_to_idx):
        """
        text: raw string
        seq_len: sequence length
        char_to_idx: dict (shared mapping)
        """
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx
        self.idx_to_char = {i: c for c, i in char_to_idx.items()}
        self.vocab_size = len(char_to_idx)

        # encode the corpus with the provided mapping (assume all chars present)
        self.data = torch.tensor(
            [self.char_to_idx[c] for c in text],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


def get_splits(batch_size=256, seq_len=50, split_ratio=0.9):
    download_data()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # build shared vocabulary from the full text (or you could use only train_text)
    char_to_idx, idx_to_char = build_vocab(text)

    N = len(text)
    split = int(N * split_ratio)
    train_text = text[:split]
    val_text = text[split:]

    train_ds = CharDataset(train_text, seq_len, char_to_idx)
    val_ds = CharDataset(val_text, seq_len, char_to_idx)

    def make_loader(dataset, shuffle):
        def collate_fn(batch):
            xs, ys = zip(*batch)
            x = torch.stack(xs, dim=0).t()  # [L, B]
            y = torch.stack(ys, dim=0).t()  # [L, B]
            return x, y

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    return train_loader, val_loader, train_ds, idx_to_char, char_to_idx


# ---------------------- generation / sampling helpers (defined once) ----------------------
def encode_string(s, char_to_idx):
    # fallback to space or the first available index if unseen
    fallback = char_to_idx.get(" ", next(iter(char_to_idx.values())))
    return [char_to_idx.get(c, fallback) for c in s]


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


def decode_sequence(idx_seq, idx_to_char):
    return "".join(idx_to_char[int(i)] for i in idx_seq)


def save_generation_file(model, idx_to_char, char_to_idx, out_path, gen_length=200, sample_config=None,
                         example_seeds=None):
    """
    sample_config: dict or None. e.g. {"temperature":0.8, "top_k":50, "top_p": None}
    example_seeds: list of strings to include (B=1 each)
    This will write:
      - greedy generation for every vocab seed
      - sampled generation for every vocab seed (if sample_config provided)
      - a few custom seed continuations (both greedy and sampled)
    """
    model.eval()
    V = len(idx_to_char)
    start_letters = torch.arange(0, V, dtype=torch.long)

    # Greedy (deterministic)
    greedy_gen = generate_greedy_batch(model, start_letters, length=gen_length)  # [T+1, V]

    # Sampled (if requested)
    sampled_gen = None
    if sample_config is not None:
        temperature = sample_config.get("temperature", 1.0)
        top_k = sample_config.get("top_k", None)
        top_p = sample_config.get("top_p", None)
        sampled_gen = generate_sampled_batch(model, start_letters, length=gen_length, temperature=temperature,
                                             top_k=top_k, top_p=top_p)

    # write to file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== GREEDY generation for every vocab seed ===\n")
        Tplus1, B = greedy_gen.shape
        for b in range(B):
            seq_indices = greedy_gen[:, b].tolist()
            seed_char = idx_to_char[seq_indices[0]]
            s = decode_sequence(seq_indices, idx_to_char)
            f.write(f"seed={repr(seed_char)} -> {s}\n")

        if sampled_gen is not None:
            f.write("\n=== SAMPLED generation for every vocab seed ===\n")
            Tplus1_s, B = sampled_gen.shape
            for b in range(B):
                seq_indices = sampled_gen[:, b].tolist()
                seed_char = idx_to_char[seq_indices[0]]
                s = decode_sequence(seq_indices, idx_to_char)
                f.write(f"seed={repr(seed_char)} -> {s}\n")

        # Add custom example seeds (B=1)
        if example_seeds is None:
            example_seeds = ["To be, or not to be, ", "thrice the branded cat hath mewed",
                             "Shall I compare thee to a summer's day?"]

        f.write("\n=== CUSTOM SEED EXAMPLES ===\n")
        for seed in example_seeds:
            # encode seed (map unknown chars to space or first char)
            seed_ids = encode_string(seed, char_to_idx)

            # Greedy continuation (preserve state across the seed)
            model.detach_states()
            model.zero_states()
            cur = torch.tensor([seed_ids[0]], dtype=torch.long, device=next(model.parameters()).device)
            generated = [cur.squeeze(0).item()]
            with torch.no_grad():
                # feed the rest of the seed (update state)
                for idx in seed_ids[1:]:
                    cur = torch.tensor([idx], dtype=torch.long, device=next(model.parameters()).device)
                    _ = model(cur)  # update state, ignore logits
                    generated.append(idx)

                # now continue (greedy)
                for _ in range(gen_length):
                    logits = model(cur)
                    cur = torch.argmax(logits, dim=-1)
                    generated.append(int(cur.item()))

            s_greedy = decode_sequence(generated, idx_to_char)

            # Sampled continuation for the same seed (reset state and feed seed again)
            if sample_config is not None:
                temperature = sample_config.get("temperature", 1.0)
                top_k = sample_config.get("top_k", None)
                top_p = sample_config.get("top_p", None)

                model.detach_states()
                model.zero_states()
                cur = torch.tensor([seed_ids[0]], dtype=torch.long, device=next(model.parameters()).device)
                generated_s = [cur.squeeze(0).item()]
                with torch.no_grad():
                    for idx in seed_ids[1:]:
                        cur = torch.tensor([idx], dtype=torch.long, device=next(model.parameters()).device)
                        _ = model(cur)
                        generated_s.append(idx)

                    for _ in range(gen_length):
                        logits = model(cur)
                        cur = sample_next_from_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                        generated_s.append(int(cur.item()))
                s_sampled = decode_sequence(generated_s, idx_to_char)
            else:
                s_sampled = "(sampling disabled)"

            f.write(f"SEED: {repr(seed)}\n")
            f.write(f"  GREEDY -> {s_greedy}\n")
            f.write(f"  SAMPLED -> {s_sampled}\n\n")

    print(f"Wrote generation output to {out_path}")


# ---------------------- build data & model ----------------------
train_dataloader, test_dataloader, train_dataset, idx_to_char, char_to_idx = get_splits(batch_size=128, seq_len=128)
vocab_size = train_dataset.vocab_size

print(f"num batches: {len(train_dataloader):,}")
print(f"vocab size: {vocab_size}")


class SNNLM(snn.TTModule):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            snn.BSRLIF(embed_dim),
            nn.Linear(embed_dim, hidden_dim, bias=False),
            snn.BSRLIF(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            snn.BSRLIF(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            snn.BSRLIF(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            snn.SRReadout(hidden_dim, beta_rank=1),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        embedded = self.emb(x)
        predicted = self.net(embedded)
        return predicted


model = SNNLM(vocab_size, 32, 512).to(device)
print(f"Num params: {model.get_param_count():,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 1

train_losses = []

# ---------------------- training loop (kept as-is per your request) ----------------------
for e in range(num_epochs):
    # TRAIN
    model.train()
    train_loss = 0
    for x, y in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        model.detach_states()
        model.zero_states()
        running_loss = 0.0
        for t in range(x.size(0)):
            input_t, target_t = x[t].to(device), y[t].to(device)

            model_output = model(input_t)
            loss = loss_fn(model_output, target_t)
            running_loss = running_loss + loss

        running_loss = running_loss / x.size(0)
        train_losses.append(running_loss.item())
        train_loss += train_losses[-1]

        optimizer.zero_grad()
        running_loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)

    # optional plot (will pop up windows if running locally)
    try:
        import matplotlib.pyplot as plt

        plt.title("LOSS")
        plt.plot(train_losses)
        plt.show()
    except Exception:
        # matplotlib might not be available in some envs; ignore if it fails
        pass

    # ---------------------- generation & save to file ----------------------
    # sample config example: temperature + top_k
    sample_cfg = {"temperature": 0.8, "top_k": 10, "top_p": None}
    out_fn = os.path.join("data", f"generated_epoch_{e}.txt")
    save_generation_file(model, idx_to_char, char_to_idx, out_fn, gen_length=400, sample_config=sample_cfg,
                         example_seeds=["To be, or not to be, ", "thrice the branded cat hath mewed",
                                        "Shall I compare thee to a summer's day?"])
    print(f"Saved generation to {out_fn}")

    # ---------------------- TEST ----------------------
    test_loss = 0
    with torch.no_grad():
        for x, y in tqdm(test_dataloader, total=len(test_dataloader), desc=f"TEST - E{e}"):
            model.detach_states()
            model.zero_states()
            running_loss = 0.0
            for t in range(x.size(0)):
                input_t, target_t = x[t].to(device), y[t].to(device)

                model_output = model(input_t)
                loss = loss_fn(model_output, target_t)
                running_loss = running_loss + loss

            running_loss = running_loss / x.size(0)
            test_loss += running_loss.item()
    test_loss /= len(test_dataloader)

    print(f"TRAIN: {train_loss:.5f} | TEST: {test_loss:.5f}")

# ---------------------- end ----------------------
