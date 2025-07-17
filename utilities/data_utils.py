import os
import torch
import numpy as np
from typing import Tuple, Dict, List
from torch.utils.data import Dataset


# def data_loading(x, batch_size, context_length, device=None):
#     """
#     Prepares input-target batches from a long sequence of tokenized data.

#     Args:
#         x (np.ndarray): 1D array of token IDs
#         batch_size (int): Number of sequences per batch
#         context_length (int): Length of each input sequence
#         device (torch.device, optional): Device to move tensors to

#     Returns:
#         Tuple[Tensor, Tensor]: (input_batch, target_batch) of shape (batch_size, context_length)
#     """
#     data_len = x.shape[0]
#     x = torch.from_numpy(x).long()

#     starts = torch.randint(0, data_len - context_length - 1, (batch_size,))
#     inputs = torch.stack([x[i:i + context_length] for i in starts])
#     targets = torch.stack([x[i + 1:i + 1 + context_length] for i in starts])

#     return inputs.to(device), targets.to(device)

class MemmapDataset(Dataset):

    def __init__(self, path: str, dtype: str, shape: List[int], context_length: int):

        self.data = np.memmap(path, dtype=dtype, mode='r', shape = tuple(shape))
        self.context_length = context_length
        self.shape = shape
      

    def __len__(self):
        """ Valid Start Positions """
        return self.shape[0] - self.context_length - 1

    def __getitem__(self, idx):

        inputs = self.data[idx : idx + self.context_length]
        targets = self.data[idx+1 : idx + 1 + self.context_length]

        return (torch.tensor(inputs), torch.tensor(targets))




def load_memmap(path, dtype, shape):
    """
    Loads memory-mapped numpy file.
    """
    return np.memmap(path, dtype=dtype, mode='r', shape=tuple(shape))


def data_loading(memmap_data, batch_size, context_length, device=None):
    """
    Yields one batch from a memmap-backed token stream.
    """
    data_len = memmap_data.shape[0]
    starts = np.random.randint(0, data_len - context_length - 1, size=batch_size)

    inputs = np.array([memmap_data[i:i + context_length] for i in starts])
    targets = np.array([memmap_data[i + 1:i + 1 + context_length] for i in starts])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets

def save_checkpoint(model, optimizer, iteration, out_path):
    """
    Saves a checkpoint containing model and optimizer state.

    Args:
        model (nn.Module): The model to save
        optimizer (Optimizer): The optimizer to save
        iteration (int): Current training iteration
        out_path (str): File path for saving checkpoint
    """
    obj = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(obj, out_path)
    print(f"[Checkpoint saved to {out_path}]")


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
    model.load_state_dict(checkpoint["model"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]