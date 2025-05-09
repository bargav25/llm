import math
import torch

def lr_schedule_cosine(t, lr_max, lr_min, t_w, t_c):
    """
    Cosine learning rate scheduler with warmup.

    Args:
        t (int): Current step
        lr_max (float): Maximum learning rate
        lr_min (float): Minimum learning rate
        t_w (int): Number of warmup steps
        t_c (int): Total number of steps

    Returns:
        float: Learning rate for current step
    """
    if t < t_w:
        return t * lr_max / t_w
    elif t <= t_c:
        return lr_min + 0.5 * (1 + math.cos((t - t_w) * math.pi / (t_c - t_w))) * (lr_max - lr_min)
    else:
        return lr_min


def gradient_clip(parameters, max_norm):
    """
    Clips gradients to prevent exploding gradient problem.

    Args:
        parameters: Iterable of model parameters
        max_norm (float): Maximum allowed norm
    """
    grads = [p.grad for p in parameters if p.grad is not None and p.requires_grad]
    total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g) for g in grads]))

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(clip_coef)


def cross_entropy_loss(predicted, target):
    """
    Numerically stable cross-entropy loss function.

    Args:
        predicted (Tensor): Logits of shape (batch_size * seq_len, vocab_size)
        target (Tensor): Target token indices of shape (batch_size * seq_len)

    Returns:
        Tensor: Scalar loss value
    """
    max_values = torch.max(predicted, dim=-1, keepdim=True).values
    logits = predicted - max_values
    log_exp_sums = torch.log(torch.sum(torch.exp(logits), dim=-1))
    target_logits = logits.gather(dim=1, index=target.unsqueeze(1)).squeeze(-1)
    loss = -target_logits + log_exp_sums
    return torch.mean(loss)