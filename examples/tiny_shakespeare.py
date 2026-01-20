import os
import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import urllib.request

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "tiny_shakespeare.txt")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


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


train_dataloader, test_dataloader, train_dataset, idx_to_char, char_to_idx = get_splits(batch_size=256, seq_len=50)
vocab_size = train_dataset.vocab_size

print(f"num batches: {len(train_dataloader):,}")
print(f"vocab size: {vocab_size}")
# print(f"vocabulary: {sorted(train_dataset.char_to_idx.keys())}")

import torch
from torch import nn
import torch as tt
from tracetorch import snn


class SNNLM(snn.TTModule):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(
            snn.BSRLIF(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            snn.BSRLIF(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            snn.BSRLIF(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            snn.BSRLIF(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            snn.Readout(emb_dim, beta_rank=1),
            nn.Linear(emb_dim, vocab_size)
        )

    def forward(self, x):
        embedded = self.emb(x)
        predicted = self.net(embedded)
        return predicted


model = SNNLM(vocab_size, 128).to(device)
print(f"Num params: {model.get_param_count():,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 10

train_losses = []

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

    import matplotlib.pyplot as plt

    plt.title("LOSS")
    plt.plot(train_losses)
    plt.show()

    # TEST AND EVAL
    model.eval()

    # EVAL
    with torch.no_grad():
        # ----- generation helpers -----
        def generate_greedy_batch(model, start_batch, length):
            """Return tensor shape [T+1, B] where T=length, each column is a generated index sequence."""
            device = next(model.parameters()).device
            seq = [start_batch.to(device)]
            model.detach_states()
            model.zero_states()
            with torch.no_grad():
                for _ in range(length):
                    logits = model(seq[-1])  # [B, vocab_size]
                    next_idx = torch.argmax(logits, dim=-1)  # [B]
                    seq.append(next_idx)
            return torch.stack(seq, dim=0)  # [T+1, B]


        def decode_sequence(idx_seq, idx_to_char):
            return "".join(idx_to_char[int(i)] for i in idx_seq)


        def save_generation_file(model, idx_to_char, out_path, gen_length=200):
            """Generate from every seed in vocab and write results to out_path."""
            model.eval()
            B = len(idx_to_char)
            start_letters = torch.arange(0, B, dtype=torch.long)
            gen = generate_greedy_batch(model, start_letters, length=gen_length)  # [T+1, B]
            Tplus1, B = gen.shape

            # write file: one generated string per line with seed char label
            with open(out_path, "w", encoding="utf-8") as f:
                for b in range(B):
                    seq_indices = gen[:, b].tolist()
                    seed_char = idx_to_char[seq_indices[0]]
                    s = decode_sequence(seq_indices, idx_to_char)
                    f.write(f"seed={repr(seed_char)} -> {s}\n")


        # After training epoch e and before test:
        out_fn = os.path.join("data", f"generated_epoch_{e}.txt")
        save_generation_file(model, idx_to_char, out_fn, gen_length=200)
        print(f"Saved generation to {out_fn}")

    # TEST
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
