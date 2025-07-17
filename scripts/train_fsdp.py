import os
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
import wandb
import yaml
import argparse
from types import SimpleNamespace
import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from models.transformer import TransformerLM
from utilities.training import lr_schedule_cosine, gradient_clip, cross_entropy_loss
from utilities.data_utils import MemmapDataset, data_loading, save_checkpoint
from optim.adamw import AdamW
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

class AppState(Stateful):
    """Wrapper for checkpointing model and optimizer with DCP"""

    def __init__(self, model, optimizer=None, step: int = 0):
        self.model = model
        self.optimizer = optimizer
        self.step = step 

    def state_dict(self):
        """ Automatically handles FSDP Sharding and state dict types"""
        # Returns in sharded format: Each rank holds its own shard of the model and optimizer states.
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)

        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "step": self.step
        }

    def load_state_dict(self, state_dict):
        # Restores state to model and optimizer
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict['model'],
            optim_state_dict=state_dict['optimizer']
        )

        self.step = state_dict['step']




# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_config(path: str) -> SimpleNamespace:
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)

def get_device(rank=0):
    return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo',
                            rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

class Trainer:

    def __init__(self, model: torch.nn.Module,
                    train_dataset: torch.utils.data.Dataset,
                    valid_dataset: torch.utils.data.Dataset,
                    train_config: SimpleNamespace,
                    optimizer_cls,
                    loss_fn,
                    rank: int):
        
        self.train_config = train_config
        torch.cuda.set_device(rank)

        self.rank = rank
        self.model = model.to(rank)
        self.train_loader = self._prepare_dataloader(train_dataset, train_config.batch_size)
        self.valid_loader = self._prepare_dataloader(valid_dataset, train_config.batch_size)

        self.loss_fn = loss_fn

        fsdp_kwargs = {}

        if train_config.use_amp:
            fsdp_kwargs['mp_policy'] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )



        # Wrap the model in FSDP
        for block in self.model.blocks:
            fully_shard(block, **fsdp_kwargs)

        self.model = fully_shard(self.model, **fsdp_kwargs)

        self.optimizer = optimizer_cls(self.model.parameters(), lr=train_config.lr_max)

        self.app_state = AppState(self.model, self.optimizer)

        # Sample check to load the model

        # self._load_checkpoint(4, "/restricted/projectnb/fhs-std-chen/bargav/temp/cs336/llm/checkpoints/")


    def _prepare_dataloader(self, dataset: torch.utils.data.Dataset, batch_size: int):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4, # Check if this is appropriate for the system
            pin_memory=True,  # What does this do? # Faster data transfer to GPU ?
            sampler=DistributedSampler(dataset) # Puts the data in the right GPU for distributed training
        )
    
    def _run_batch(self, inputs, targets, train: bool = True):
        """Run a single training batch, with Automatic Mixed Precision (AMP)."""

        logits = self.model(inputs).view(-1, self.train_config.vocab_size) # Reshape logits to match targets
        loss = self.loss_fn(logits, targets.view(-1))  # Flatten targets to match logits

        if train:
            self.optimizer.zero_grad()  
            loss.backward()
            gradient_clip(self.model.parameters(), self.train_config.clip_grad)  # Clip gradients to prevent exploding gradients
            self.optimizer.step()

        return loss.item()  # Return the loss value for logging

    
    def _save_checkpoint(self, step: int, output_dir: str):
        """Saving checkpoint using DCP"""
        # Update state with current step
        self.app_state.step = step

        checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}")

        # All ranks participate in the DCP
        state_dict = {"app": self.app_state}

        dist.barrier() # Ensuring Sync

        dcp.save(state_dict, checkpoint_id = checkpoint_path)

        print(f"Model Saved at {step}")

    def _load_checkpoint(self, step: int, output_dir: str):
        """Saving checkpoint using DCP"""
        # Update state with current step
        self.app_state.step = step

        checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}")

        # All ranks participate in the DCP

        state_dict = {"app": self.app_state}

        dcp.load(
            state_dict=state_dict,
            checkpoint_id=checkpoint_path,
        )

        dist.barrier() # Ensuring Sync, because of print statement

        print(f"Model Loaded for step: {step}")

    def train(self, num_steps: int = None, output_dir: str = None):
        """Train the model for a specified number of steps."""

        num_steps = num_steps if num_steps is not None else self.train_config.num_steps
        output_dir = output_dir if output_dir is not None else self.train_config.output_dir 

        train_iter = iter(self.train_loader)
        val_iter = iter(self.valid_loader)

        for step in range(num_steps):

            # Run training step

            self.model.train()

            lr = lr_schedule_cosine(step, self.train_config.lr_max, self.train_config.lr_min, 
                                self.train_config.warmup_steps, num_steps)

            for g in self.optimizer.param_groups:
                g['lr'] = lr

            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                # Restart from the first batch if done
                train_iter = iter(self.train_loader)
                inputs, targets = next(train_iter)

            inputs, targets = inputs.to(self.rank), targets.to(self.rank)


            train_loss = self._run_batch(inputs, targets, train=True)

            if step % self.train_config.eval_interval == 0:

                # Run Validation step

                self.model.eval()

                try:
                    inputs, targets = next(val_iter)
                except:
                    # Restart from the first batch if done
                    val_iter = iter(self.valid_loader)
                    inputs, targets = next(val_iter)

                inputs, targets = inputs.to(self.rank), targets.to(self.rank)

                val_loss = self._run_batch(inputs, targets, train=False)

                dist.barrier()

                print(f"Step: {step} | Rank: {self.rank} | Training Loss: {train_loss:.3f} | Validation Loss: {val_loss:.3f}")

            
            if step % self.train_config.ckpt_interval == 0:

                # Save the FSDP Checkpoint
                self._save_checkpoint(step, output_dir)


def train(rank, world_size, train_config):
    """Main training function for distributed training"""

    # Setup distributed training
    setup(rank, world_size)

    # Load datasets
    train_dataset = MemmapDataset(train_config.train_memmap_path, train_config.dtype, train_config.data_shape, train_config.context_length)
    valid_dataset = MemmapDataset(train_config.val_memmap_path , train_config.dtype, train_config.data_shape, train_config.context_length)

    # Initialize model
    model = TransformerLM(
        vocab_size=train_config.vocab_size,
        context_length=train_config.context_length,
        d_model=train_config.d_model,
        num_layers=train_config.num_layers,
        num_heads=train_config.num_heads,
        d_ff=train_config.d_ff,
        rope_theta=train_config.rope_theta,
        device=None
    ) # Is it getting loaded in CPU ?


    # Initialize loss function
    loss_fn = cross_entropy_loss

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer_cls=AdamW,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        rank=rank,
        train_config=train_config
    )

    # Start training
    trainer.train()

    cleanup()


def main():

    # ✅ Safeguards to prevent watchdog hangs
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "900"  # Optional: 15min if not disabling
    os.environ["NCCL_P2P_DISABLE"] = "1"  # Avoid some known deadlocks
    os.environ["NCCL_ALGO"] = "Ring"      # More stable for multi-GPU
    os.environ["NCCL_DEBUG"] = "INFO"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.use_wandb = args.use_wandb

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()