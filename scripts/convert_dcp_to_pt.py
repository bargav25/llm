import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

import yaml
from types import SimpleNamespace
from models.transformer import TransformerLM


class AppState(Stateful):
    def __init__(self, model, step=0):
        self.model = model
        self.step = step

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, [])
        return {"model": model_state_dict, "step": self.step}

    def load_state_dict(self, state_dict):
        set_state_dict(self.model, [], model_state_dict=state_dict["model"], optim_state_dict=[])
        self.step = state_dict["step"]


def load_config(path: str) -> SimpleNamespace:
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo',
                            rank=rank, world_size=world_size)


def convert_dcp_to_pt(cfg_path, dcp_path, output_path):
    # Setup distributed
    torch.cuda.set_device(0)
    setup(0, 1)

    # Load config
    cfg = load_config(cfg_path)
    device = torch.device("cuda:0")

    # Build model
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

    # Rewrap FSDP exactly as in training
    fsdp_kwargs = {}
    if getattr(cfg, "use_amp", False):
        fsdp_kwargs['mp_policy'] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

    for block in model.blocks:
        fully_shard(block, **fsdp_kwargs)
    model = fully_shard(model, **fsdp_kwargs)

    # Load from DCP
    app_state = AppState(model)
    dcp.load({"app": app_state}, checkpoint_id=dcp_path)
    print(f"✅ Loaded FSDP+DCP checkpoint from: {dcp_path}")

    # Extract full unsharded state_dict
    full_state_dict = model.state_dict()

    # Strip module prefix and convert DTensors to regular tensors
    stripped_state_dict = {}
    for k, v in full_state_dict.items():
        # Remove module prefix if present
        key = k.replace("module.", "")
        
        # Convert DTensor to regular tensor if needed
        if hasattr(v, 'to_local'):  # DTensor has to_local method
            tensor = v.to_local()
        elif hasattr(v, '_local_tensor'):  # Alternative DTensor attribute
            tensor = v._local_tensor
        else:
            tensor = v
        
        stripped_state_dict[key] = tensor

    torch.save({"model_state_dict": stripped_state_dict}, output_path)
    print(f"✅ Converted and saved to: {output_path}")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--dcp_path", type=str, required=True, help="Path to DCP checkpoint (e.g., checkpoint_step_0)")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the .pt checkpoint")
    args = parser.parse_args()

    convert_dcp_to_pt(args.cfg, args.dcp_path, args.output_path)