import argparse
import math
import numpy as np
import torch
from torch import nn
from tqdm import trange
import wandb
import os
import yaml
from types import SimpleNamespace

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.transformer import TransformerLM
from utilities.training import lr_schedule_cosine, gradient_clip, cross_entropy_loss
from utilities.data_utils import load_memmap, data_loading, save_checkpoint
from optim.adamw import AdamW


def load_config(path: str) -> SimpleNamespace:
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)

def main():
    parser = argparse.ArgumentParser(description="Train TransformerLM with memory-mapped data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_wandb:
        wandb.init(project="transformer-lm", config=vars(cfg))

    # Load training and validation memory-mapped data
    train_data = load_memmap(cfg.train_memmap_path, cfg.dtype, cfg.data_shape)
    val_data = load_memmap(cfg.val_memmap_path, cfg.dtype, cfg.data_shape)

    # Initialize model
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

    optimizer = AdamW(model.parameters(), lr=cfg.lr_max)
    loss_fn = cross_entropy_loss

    # Training loop
    for step in trange(cfg.num_steps):
        model.train()
        lr = lr_schedule_cosine(step, cfg.lr_max, cfg.lr_min, cfg.warmup_steps, cfg.num_steps)
        for g in optimizer.param_groups:
            g['lr'] = lr

        inputs, targets = data_loading(train_data, cfg.batch_size, cfg.context_length, device)
        logits = model(inputs).view(-1, cfg.vocab_size)
        loss = loss_fn(logits, targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        gradient_clip(model.parameters(), cfg.clip_grad)
        optimizer.step()

        # Periodic validation and logging
        if step % cfg.eval_interval == 0 or step == cfg.num_steps - 1:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = data_loading(val_data, cfg.batch_size, cfg.context_length, device)
                val_logits = model(val_inputs).view(-1, cfg.vocab_size)
                val_loss = loss_fn(val_logits, val_targets.view(-1))

            print(f"[{step}] Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | LR: {lr:.6f}")

            if args.use_wandb:
                wandb.log({"train_loss": loss.item(), "val_loss": val_loss.item(), "lr": lr}, step=step)

            save_checkpoint(model, optimizer, step, cfg.save_path)

if __name__ == "__main__":
    main()