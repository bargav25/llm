import torch
import torch.nn.functional as F
from models.layers import softmax
from models.tokenizer import BPETokenizer
from utilities.data_utils import load_checkpoint
import yaml
import os
from models.transformer import TransformerLM

from types import SimpleNamespace

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_checkpoint(src_path, model, optimizer=None):
    """
    Loads model and optionally optimizer state from a checkpoint.

    Args:
        src_path (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (Optimizer, optional): Optimizer to restore

    Returns:
        int: Iteration number from the checkpoint
    """
    checkpoint = torch.load(src_path)
    state_dict = checkpoint.get('model_state_dict', checkpoint)  # handles both flat and nested
    stripped_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(stripped_state_dict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return 

def load_config(path: str) -> SimpleNamespace:
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)

@torch.no_grad()
def decode(prompt_text, model, tokenizer, temperature=1.0, top_p=0.9, max_new_tokens=200):
    model.eval()
    token_ids = tokenizer.encode(prompt_text)
    token_ids = torch.tensor([token_ids], dtype=torch.long).to(next(model.parameters()).device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(token_ids)[0]  # (batch, seq_len, vocab_size)

        next_token_logits = logits[-1, :]  # get last token's logits

        # Apply temperature
        logits_scaled = next_token_logits / temperature

        # Convert to probabilities
        probs = F.softmax(logits_scaled, dim=-1)

        # Top-p (nucleus) filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        # Sample
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token_id = sorted_indices.gather(-1, next_token)


        # Append new token
        token_ids = torch.cat([token_ids[0], next_token_id ], dim=-1).unsqueeze(dim=0)

        # Decode and check
        decoded = tokenizer.decode(next_token_id.tolist())
        if decoded == "<|endoftext|>":
            break

        print(decoded, end='', flush=True)

    print("\n\n--- END ---")

if __name__ == "__main__":

    tokenizer = BPETokenizer.from_file("/restricted/projectnb/fhs-std-chen/bargav/temp/cs336/bpe/output/bpe_tokenizer.pkl")

    cfg = load_config("/restricted/projectnb/fhs-std-chen/bargav/temp/cs336/llm/configs/train_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        device=device
    ).to(device)

    load_checkpoint("/restricted/projectnb/fhs-std-chen/bargav/temp/cs336/llm/checkpoints_fsdp/fsdp_converted.pt",model)

    decode(
        prompt_text="Once upon a time",
        model=model,
        tokenizer=tokenizer,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100
    )