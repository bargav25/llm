import torch 
from torch import nn
from einops import einsum

class Linear(nn.Module):
    """
    A custom linear (fully connected) layer implemented using einsum instead of torch.matmul.
    Projects input of shape (..., in_features) to (..., out_features).
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.W = torch.empty((out_features, in_features), device=device, dtype=dtype)
        self.W = nn.Parameter(self.W, requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(self.W)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Applies: Y = X @ W.T using einsum notation
        return einsum(X, self.W, "... n, m n -> ... m")


class Embedding(nn.Module):
    """
    Standard embedding layer mapping token IDs to dense vectors of fixed size.
    """

    def __init__(self, num_vocab, embed_dim, device=None, dtype=None):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_vocab, embed_dim, device=device, dtype=dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize embeddings using truncated normal distribution
        nn.init.trunc_normal_(self.embeddings)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids (Tensor): Tensor of token IDs with arbitrary shape.
        Returns:
            Tensor: Embedding vectors of shape (..., embed_dim)
        """
        return self.embeddings[token_ids]


class SWIGLU(nn.Module):
    """
    SwiGLU feedforward layer:
        output = W2 * ( (X @ W1) * sigmoid(X @ W1) * (X @ W3) )
    A variation of the Gated Linear Unit with element-wise gating and non-linearity.
    """

    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # W1 and W3 for gated activation, W2 for projecting back to d_model
        self.W1 = nn.Parameter(torch.empty(self.d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(self.d_model, d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        a = einsum(X, self.W1, "... d_model, d_ffn d_model -> ... d_ffn")
        b = einsum(X, self.W3, "... d_model, d_ffn d_model -> ... d_ffn")

        a = a * torch.sigmoid(a)
        h = a * b

        return einsum(h, self.W2, "... d_ffn, d_model d_ffn -> ... d_model")


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Normalizes input based on RMS across last dimension without subtracting the mean.
    Applies a learned gain after normalization.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize gain vector to ones
        nn.init.ones_(self.gain)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (Tensor): Input tensor of shape (..., d_model)
        Returns:
            Tensor: RMS-normalized tensor of same shape
        """
        rms = torch.sqrt(torch.sum(X ** 2, dim=-1, keepdim=True) / self.d_model + self.eps)
        return (X / rms) * self.gain


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax along a specified dimension.

    Args:
        x (Tensor): Input tensor.
        dim (int): Dimension along which softmax is computed.

    Returns:
        Tensor: Softmax-normalized tensor.
    """
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - max_val)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)