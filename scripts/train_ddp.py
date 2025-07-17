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
                    optimizer: torch.optim.Optimizer,
                    loss_fn: callable,
                    rank: int):
        
        self.train_config = train_config
        torch.cuda.set_device(rank)

        self.rank = rank
        self.model = model.to(rank)
        self.train_loader = self._prepare_dataloader(train_dataset, train_config.batch_size)
        self.valid_loader = self._prepare_dataloader(valid_dataset, train_config.batch_size)

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.scaler = torch.amp.GradScaler()  if self.train_config.use_amp else None # Scales gradients to prevent underflow in mixed precision training


        # Wrap the model in DDP
        self.model = DDP(self.model, device_ids=[rank] if torch.cuda.is_available() else None)
        
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

        with torch.amp.autocast(enabled=self.train_config.use_amp, device_type="cuda"):
            logits = self.model(inputs).view(-1, self.train_config.vocab_size) # Reshape logits to match targets
            loss = self.loss_fn(logits, targets.view(-1))  # Flatten targets to match logits

        if train:
            self.optimizer.zero_grad()  # Zero the gradients before backward pass
            # Backward pass with AMP
            self.scaler.scale(loss).backward()  if self.scaler else loss.backward()
            gradient_clip(self.model.parameters(), self.train_config.clip_grad)  # Clip gradients to prevent exploding gradients
            self.scaler.step(self.optimizer)  if self.scaler else self.optimizer.step()
            self.scaler.update()  if self.scaler else None  # Update the scaler for next iteration (AMP)

        return loss.item()  # Return the loss value for logging

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        """Run a single epoch of training or validation."""
        total_loss = 0.0
        self.model.train() if train else self.model.eval()  # Set model to training or evaluation mode

        step_type = "Training" if train else "Validation"

        if train:
            dataloader.sampler.set_epoch(epoch) # What it does ? # Ensures that the data is shuffled differently at each epoch

        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.rank), targets.to(self.rank) # Move data to the appropriate device
            batch_loss = self._run_batch(inputs, targets, train)
            total_loss += batch_loss

            print(f"Rank {self.rank}, Epoch: {epoch}, Batch {i+1}/{len(dataloader)},  {step_type} Loss: {batch_loss:.4f}")

        avg_loss = total_loss / len(dataloader)  # Average loss over the epoch
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, output_dir: str):

        obj = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
         }
        
        if self.scaler:
            obj['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(obj, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"))

    def train(self, num_epochs: int = None, output_dir: str = None):
        """Train the model for a specified number of epochs."""

        num_epochs = num_epochs if num_epochs is not None else self.train_config.num_epochs
        output_dir = output_dir if output_dir is not None else self.train_config.output_dir

        
        for epoch in range(num_epochs):

            # Run training epoch
            train_loss = self._run_epoch(epoch, self.train_loader, train=True)
            print(f"Rank {self.rank}, Epoch: {epoch}, Training Loss: {train_loss:.4f}")

            # Run Validation epoch
            val_loss = self._run_epoch(epoch, self.valid_loader, train=False)
            print(f"Rank {self.rank}, Epoch: {epoch}, Validation Loss: {val_loss:.4f}")

            # Save checkpoint
            if self.rank == 0: # Since the parameters are synchronized across all processes, we only save the checkpoint from rank 0
                self._save_checkpoint(epoch, output_dir)


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

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.lr_max
    )
    # Initialize loss function
    loss_fn = cross_entropy_loss


    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        rank=rank,
        train_config=train_config
    )

    # Start training
    trainer.train()


def main():

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