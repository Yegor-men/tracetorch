"""Train a residual byte-level SNN on Alice's Adventures in Wonderland.

Run: python examples/alice_language_model/main.py --device cpu
Pass --corpus PATH to use another uncompressed byte corpus, including enwik8.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

import tracetorch as tt


EXAMPLE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXAMPLE_ROOT.parent / "data"
CORPUS_URL = "https://www.gutenberg.org/ebooks/11.txt.utf-8"


def load_corpus(corpus_path, max_bytes):
    if corpus_path is None:
        corpus_path = DATA_ROOT / "alice.txt"
        if not corpus_path.exists():
            corpus_path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(CORPUS_URL, timeout=60) as response:
                content = response.read()
            corpus_path.write_bytes(content)
        content = corpus_path.read_bytes()
        # Keep Gutenberg's source file intact; train on the book, not its licence header/footer.
        start_marker = b"*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = b"*** END OF THE PROJECT GUTENBERG EBOOK"
        start = content.find(start_marker)
        end = content.find(end_marker)
        if start >= 0:
            content = content[content.find(b"\n", start) + 1:end if end >= 0 else None]
    else:
        # Read only the requested budget when using a larger local corpus.
        with corpus_path.open("rb") as source:
            content = source.read(max_bytes if max_bytes else -1)
    if max_bytes:
        content = content[:max_bytes]
    return torch.tensor(list(content), dtype=torch.long)


def make_streams(tokens, batch_size):
    """Split contiguous text into fixed lanes, returning [T, B] byte indices."""
    steps = tokens.numel() // batch_size
    if steps < 2:
        raise ValueError("Corpus split must contain at least two bytes per stream lane")
    return tokens[:steps * batch_size].view(batch_size, steps).T.contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lin1 = nn.Linear(width, width)
        self.neuron = tt.snn.LIB(
            width, beta=torch.rand(width), spike_fn=tt.snn.spike_fn.smooth,
        )
        self.lin2 = nn.Linear(width, width)
        # Small nonzero projections start near the identity and let both projections learn.
        for projection in (self.lin1, self.lin2):
            nn.init.normal_(projection.weight, std=0.01)
            nn.init.zeros_(projection.bias)

    def forward(self, x):
        return x + self.lin2(self.neuron(self.lin1(x)))


class ByteSNN(tt.Model):
    def __init__(self, width=64, depth=2):
        super().__init__()
        self.embedding = nn.Embedding(256, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(depth)))
        self.readout = nn.Linear(width, 256)

    def forward(self, byte):
        return self.readout(self.blocks(self.embedding(byte)))


def run_epoch(model, streams, optimizer=None):
    training = optimizer is not None
    model.train(training)
    model.reset_states()
    total_loss = total_bytes = 0
    with torch.set_grad_enabled(training):
        for timestep in range(streams.shape[0] - 1):
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(streams[timestep])
            targets = streams[timestep + 1]
            loss = F.cross_entropy(logits, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # Preserve stream memory without backpropagating into earlier updates.
                model.detach_states()
            total_loss += loss.item() * targets.numel()
            total_bytes += targets.numel()
    model.reset_states()
    return total_loss / total_bytes


def generate(model, prompt, length, device, temperature=0.8):
    model.eval()
    model.reset_states()
    result = bytearray(prompt.encode("utf-8"))
    if not result:
        raise ValueError("Generation prompt must contain at least one byte")
    with torch.no_grad():
        for byte in result:
            logits = model(torch.tensor([byte], device=device))
        for _ in range(length):
            probabilities = (logits / temperature).softmax(-1)
            next_byte = torch.multinomial(probabilities, 1).squeeze(-1)
            result.append(next_byte.item())
            logits = model(next_byte)
    model.reset_states()
    return result.decode("utf-8", errors="replace")


def restore_continuation(model, checkpoint, device):
    """Save prefix states and reproduce every output in the remaining sequence."""
    model.eval()
    model.reset_states()
    sequence = torch.tensor(list(b"Alice was beginning to get very tired."), device=device).unsqueeze(1)
    cursor = 16
    with torch.no_grad():
        for byte in sequence[:cursor]:
            model(byte)
        stream = {"states": model.save_states(), "cursor": cursor, "sequence": sequence.cpu()}
        torch.save(stream, checkpoint.parent / "states.pt")
        expected = torch.stack([model(byte) for byte in sequence[cursor:]])
        saved_model = torch.load(checkpoint, map_location=device, weights_only=True)
        restored = ByteSNN(saved_model["width"], saved_model["depth"]).to(device).eval()
        restored.load_state_dict(saved_model["weights"])
        saved = torch.load(checkpoint.parent / "states.pt", map_location=device, weights_only=True)
        restored.load_states(saved["states"])
        actual = torch.stack([restored(byte) for byte in saved["sequence"][saved["cursor"]:]])
    torch.testing.assert_close(actual, expected)
    model.reset_states()
    return (actual - expected).abs().max().item()


def plot_history(history):
    fig, axis = plt.subplots(figsize=(7, 3.5), layout="constrained")
    axis.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Training")
    axis.plot(history["validation_loss"], label="Validation")
    axis.set_xlabel("Epoch (0 = before training)")
    axis.set_ylabel("Loss (nats per byte)")
    axis.legend()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus", type=Path, help="Local uncompressed text/byte file")
    parser.add_argument("--max-bytes", type=int, default=40000, help="0 reads the whole corpus")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--checkpoint", type=Path, default=EXAMPLE_ROOT / "checkpoints" / "model.pt")
    parser.add_argument("--prompt", default="Alice ")
    parser.add_argument("--generate-bytes", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=EXAMPLE_ROOT / "figures")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.width, args.depth, args.threads) <= 0:
        parser.error("epochs, batch size, width, depth and threads must be positive")
    if min(args.max_bytes, args.generate_bytes) < 0 or args.learning_rate <= 0 or not args.prompt:
        parser.error("byte limits must be nonnegative, learning rate positive, and prompt nonempty")
    return args


def main():
    args = parse_args()
    plt.style.use("default")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tokens = load_corpus(args.corpus, args.max_bytes)
    # Separate text regions prevent validation from sharing training targets.
    split = int(tokens.numel() * 0.9)
    train = make_streams(tokens[:split], args.batch_size).to(args.device)
    validation = make_streams(tokens[split:], args.batch_size).to(args.device)
    model = ByteSNN(args.width, args.depth).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"Byte SNN: {tokens.numel()} bytes; {args.batch_size} streams; {args.device}")
    baseline = run_epoch(model, validation)
    print(f"Before training: validation loss={baseline:.4f} nats/byte")
    history = {"train_loss": [], "validation_loss": [baseline]}
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train, optimizer)
        validation_loss = run_epoch(model, validation)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        print(f"Epoch {epoch}: train loss={train_loss:.4f}; validation loss={validation_loss:.4f} nats/byte")
    runtime = perf_counter() - started
    print(f"Training and evaluation: {runtime:.1f}s (download excluded)")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "weights": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "width": args.width,
        "depth": args.depth,
        "completed_epochs": args.epochs,
    }, args.checkpoint)
    print(f"Saved epoch-boundary checkpoint: {args.checkpoint}")
    print(generate(model, args.prompt, args.generate_bytes, args.device))
    difference = restore_continuation(model, args.checkpoint, args.device)
    print(f"Restored suffix matches uninterrupted output; maximum difference={difference:.3g}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "history.json").write_text(json.dumps(
        {**history, "runtime_seconds": runtime}, indent=2,
    ))
    plot_history(history).savefig(args.output_dir / "learning.png", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    main()
