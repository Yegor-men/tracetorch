import torch
from torch import nn
import tracetorch
from tracetorch import snn
from safetensors.torch import load_file
import os


def generate_text(model, seed_bytes, gen_length, sample_config, device):
    model.zero_states()
    cur = torch.tensor([seed_bytes[0]], dtype=torch.long, device=device)
    generated = [cur.item()]

    # Process seed bytes
    for b in seed_bytes[1:]:
        _ = model(cur)
        generated.append(b)
        cur = torch.tensor([b], dtype=torch.long, device=device)

    # Generate new tokens
    is_greedy = sample_config["top_k"] == 1
    for _ in range(gen_length):
        logits = model(cur)
        if is_greedy:
            cur = torch.argmax(logits, dim=-1)
        else:
            cur = sample_next_from_logits(logits, sample_config)
        generated.append(cur.item())

    generated_str = bytes(generated).decode("utf-8", errors="replace")
    return generated_str


def sample_next_from_logits(logits, sample_config):
    """
    logits: [B, V]
    returns: next indices [B]
    """
    temperature = sample_config["temperature"]
    top_k = sample_config["top_k"]
    top_p = sample_config["top_p"]

    # top-k: zero out everything not in top_k by setting to -inf
    if top_k is not None and top_k > 0:
        top_k_val = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, top_k_val, dim=-1)  # [B, top_k_val]
        min_vals = vals[..., -1].unsqueeze(-1)  # [B, 1]
        logits = torch.where(logits < min_vals, torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
                             logits)

    # top-p (nucleus) - vectorized
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = nn.functional.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for tokens to remove (cumulative prob > top_p)
        remove_mask = cum_probs > top_p
        # Never remove the first token (ensure at least one remains)
        remove_mask[..., 0] = False

        # Apply mask to logits by setting to -inf
        for i in range(logits.size(0)):
            logits[i, sorted_idx[i][remove_mask[i]]] = float('-inf')

    probs = nn.functional.softmax(logits, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_idx


def save_generation_file(model, out_path, gen_length, sample_cfg, device, example_seeds=None):
    if example_seeds is None:
        example_seeds = [
            "Skibidi Toilet (stylized as skibidi toilet) is an animated web series created by",
            "Walter Hartwell White, also known by his alias Heisenberg",
            'Gustavo "Gus" Fring (Spanish pronunciation: [gusˈtaβo ˈfɾin]) is a fictional character portrayed by Giancarlo Esposito in the Breaking Bad crime drama franchise.'
        ]

    greedy_cfg = {"temperature": 0.5, "top_k": 1, "top_p": 0.9}
    sample_cfg = {"temperature": 0.5, "top_k": 20, "top_p": 0.9} if sample_cfg is None else sample_cfg

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:

        for seed_str in example_seeds:
            f.write(f"SEED: {repr(seed_str)}\n")
            seed_bytes = list(seed_str.encode("utf-8"))
            greedy_str = generate_text(model, seed_bytes, gen_length, greedy_cfg, device)
            sample_str = generate_text(model, seed_bytes, gen_length, sample_cfg, device)
            f.write(f"  GREEDY  → {greedy_str}\n")
            f.write(f"  SAMPLED → {sample_str}\n\n")

    print(f"Wrote generation output to {out_path}")


def sample(model, name: str, device, sample_cfg=None):
    sample_cfg = {"temperature": 0.5, "top_k": 20, "top_p": 0.9} if sample_cfg is None else sample_cfg
    out_path = os.path.join("samples", f"{name}.txt")
    save_generation_file(model, out_path, gen_length=1000, sample_cfg=sample_cfg, device=device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from architecture import SNNLM

    model = SNNLM(2048, 10).to(device)
    modelfile = "checkpoints/step_20300_e1_bpb1691.safetensors"
    model.load_state_dict(load_file(modelfile))

    model.eval()
    with torch.no_grad():
        sample(model, "sample_gen", device)
