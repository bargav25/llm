import math 
import torch
from torch import nn
from einops import einsum, rearrange
from layers import softmax, Linear

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None, device=None, dtype=None) -> torch.Tensor:
    """
    Computes scaled dot-product attention.

    Args:
        q (Tensor): Query tensor of shape (batch_size, num_heads, seq_len, d_k)
        k (Tensor): Key tensor of shape (batch_size, num_heads, seq_len, d_k)
        v (Tensor): Value tensor of shape (batch_size, num_heads, seq_len, d_v)
        mask (Tensor, optional): Causal or padding mask. Shape can be (seq_len, seq_len) or (batch_size, seq_len, seq_len)
        device (torch.device, optional): Target device for computation.
        dtype (torch.dtype, optional): Target data type for computation.

    Returns:
        Tensor: Output tensor of shape (batch_size, num_heads, seq_len, d_v)
    """
    d_k = q.size(-1)

    if device or dtype:
        q = q.to(device=device, dtype=dtype)
        k = k.to(device=device, dtype=dtype)
        v = v.to(device=device, dtype=dtype)

    # Compute attention scores
    attention = einsum(q, k, "... i d_k, ... j d_k -> ... i j")
    attention /= math.sqrt(d_k)  # Scale scores

    # Apply mask if present
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # Expand for batch and heads
        mask = mask.to(dtype=bool, device=device)
        attention = attention.masked_fill(~mask, float("-inf"))

    # Softmax normalization
    softmax_values = softmax(attention, dim=-1)

    # Weighted sum of values
    output = einsum(softmax_values, v, "... i j, ... j d_v -> ... i d_v")
    return output


class ROPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Uses sinusoidal embeddings to rotate input vectors in the complex plane, enabling positional encoding without explicit addition.

    Precomputes sin and cos values for efficiency, and rotates Q/K vectors during attention.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Inverse frequency vector
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device).float() / d_k)

        # Position indices and angle matrix
        positions = torch.arange(max_seq_len, device=device).float()
        angles = einsum(positions, inv_freq, "p, d -> p d")
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)

        # Precompute sine and cosine values
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the dimensions by 90 degrees (for RoPE)
        """
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack([-x_odd, x_even], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to input.

        Args:
            x (Tensor): Input tensor of shape (..., seq_len, d_k)
            token_positions (Tensor): Position indices for each token (..., seq_len)

        Returns:
            Tensor: RoPE-transformed input of same shape
        """
        assert x.shape[-1] == self.d_k
        seq_len = x.shape[-2]
        assert seq_len <= self.max_seq_len

        # Get sine and cosine values for given positions
        if token_positions is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[token_positions]
            sin = self.sin_cached[token_positions]

        x_rot = self.rotate_half(x)
        return einsum(x, cos, "... s d, ... s d -> ... s d") + einsum(x_rot, sin, "... s d, ... s d -> ... s d")


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention module (without positional encoding).
    Projects input to Q, K, V spaces, computes attention, and projects back.
    """

    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.device = device

        # Linear projections for Q, K, V
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)

        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Output of shape (batch_size, seq_len, d_model)
        """
        seq_len = X.size(1)

        q = rearrange(self.W_q(X), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.W_k(X), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.W_v(X), "b s (h d) -> b h s d", h=self.num_heads)

        # Causal mask for autoregressive decoding
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))

        attended = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        out = rearrange(attended, "b h s d -> b s (h d)")

        return self.out_proj(out)


class RopeMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Positional Embeddings applied to Q and K.
    Useful for models like GPT-NeoX and LLaMA.
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.device = device

        self.rope = ROPE(theta, self.d_k, max_seq_len, device=device, dtype=dtype)

        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)

        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, X: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            X (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            token_positions (Tensor): Optional position indices for RoPE

        Returns:
            Tensor: Output of shape (batch_size, seq_len, d_model)
        """
        seq_len = X.size(1)

        q = rearrange(self.W_q(X), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.W_k(X), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.W_v(X), "b s (h d) -> b h s d", h=self.num_heads)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))

        attended = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        out = rearrange(attended, "b h s d -> b s (h d)")

        return self.out_proj(out)