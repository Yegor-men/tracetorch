import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class SNNLM(snn.TTModule):
    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()

        self.emb = nn.Embedding(256, embed_dim)

        layers = [nn.Linear(embed_dim, hidden_dim)]
        for _ in range(num_layers):
            layers.append(snn.BSRLIF(hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        layers.append(snn.SRReadout(hidden_dim, beta_rank=1))
        layers.append(nn.Linear(hidden_dim, 256))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        embedded_byte = self.emb(x)
        predicted_byte = self.net(embedded_byte)
        return predicted_byte


model = SNNLM(256, 1024, 5).to(device)
print(f"Num params: {model.get_param_count():,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-5)
loss_fn = nn.CrossEntropyLoss()

from tqdm import tqdm
import time

from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import random


class ConcatByteDataset(Dataset):
    def __init__(self, split="train", max_samples=None, seq_len=256):
        # Load dataset (ag_news for POC; switch to "bookcorpus" for general text)
        ds = load_dataset("ag_news", split=split)
        if max_samples:
            ds = ds.select(range(max_samples))

        # Concat all texts with optional separator
        sep = list("\n\n---\n\n".encode('utf-8'))  # Optional boundary
        all_bytes = []
        for item in ds:
            text = item['text']  # For ag_news; use 'text' for bookcorpus
            all_bytes += list(text.encode('utf-8')) + sep
        self.data = torch.tensor(all_bytes, dtype=torch.long)  # One big tensor

        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len  # Num chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1]  # +1 for target
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


s = time.time()
dataset = ConcatByteDataset(max_samples=1000, seq_len=256)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Dataset loading finished in {time.time() - s:.2f}s")

import torch.nn.functional as F
from tqdm import tqdm
import os


def sample_next(logits, top_k=10, top_p=0.95, temperature=1.0):
    """
    logits: [batch, vocab]  (here [1, 256])
    Supports top-k and top-p (nucleus).
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filter
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
        threshold = top_k_vals[..., -1, None]  # [1,1]
        logits = torch.where(logits < threshold, torch.full_like(logits, -float('inf')), logits)

    # Top-p filter (on remaining)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Shift mask to keep at least the top token
        sorted_mask = cum_probs > top_p
        sorted_mask[..., 0] = False
        # Back to original indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, sorted_mask)
        logits[mask] = -float('inf')

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch=1] → scalar, but keep as [1]
    return next_token


def generate_and_save(model, steps, max_gen_len=1000, top_k=10, top_p=0.95, temperature=1.0, output_dir="."):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Seed with <START> bytes
    start_str = "<START>"
    start_bytes = list(start_str.encode('utf-8'))
    end_str = "<END>"
    end_bytes = list(end_str.encode('utf-8'))
    end_len = len(end_bytes)

    # Generate
    model.zero_states()
    generated = start_bytes[:]
    cur_input = torch.tensor([start_bytes[0]], dtype=torch.long, device=device)  # [1]
    with torch.no_grad():
        # Prime with start seq (except first)
        for byte_id in start_bytes[1:]:
            _ = model(cur_input)
            generated.append(byte_id)
            cur_input = torch.tensor([byte_id], dtype=torch.long, device=device)  # [1]

        # Generate loop
        for _ in tqdm(range(max_gen_len), desc="Generating"):
            logits = model(cur_input)  # [1, 256]
            next_id = sample_next(logits, top_k=top_k, top_p=top_p, temperature=temperature)
            generated.append(int(next_id.item()))
            cur_input = next_id.unsqueeze(0) if next_id.dim() == 0 else next_id  # Ensure [1]

            # Check for <END> in rolling window
            if len(generated) >= end_len and generated[-end_len:] == end_bytes:
                break

    # Decode and save text
    gen_text = bytes(generated).decode('utf-8', errors='replace')
    text_path = os.path.join(output_dir, f"gen_step_{steps}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(gen_text)

    # Save model
    model_path = os.path.join(output_dir, f"model_step_{steps}.pt")
    torch.save(model.state_dict(), model_path)

    model.train()  # Back to train mode
    print(f"Saved generation to {text_path} and model to {model_path}")


num_epochs = 100
losses = []
for e in range(num_epochs):
    model.train()
    for i, (batch_x, batch_y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Stage 1 - E{e}"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # [batch, seq_len]
        model.zero_states()  # Reset per batch
        logits = []
        for t in range(batch_x.size(1)):  # Stepwise over seq (if SNN needs it; batch dim preserved)
            out = model(batch_x[:, t])  # [batch]
            logits.append(out)
        logits = torch.stack(logits, dim=1)  # [batch, seq, 256]
        loss = loss_fn(logits.view(-1, 256), batch_y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    generate_and_save(model, e)
    plt.title('Training Loss Over Time')
    plt.plot(losses, label='Running Loss')
    plt.xlabel('Article Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
