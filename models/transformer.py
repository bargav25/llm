import torch
from torch import nn 
from models.attention import RopeMultiHeadAttention
from models.layers import SWIGLU, RMSNorm, Embedding, Linear


class TransformerBlock(nn.Module):
    """
    A single transformer block that includes:
      - Pre-normalized RoPE-based multi-head self-attention
      - SwiGLU-based feedforward network
      - Residual connections around both components
    """

    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, device=None, dtype=None):
        super().__init__()

        # Rotary-encoded self-attention
        self.rope_attention = RopeMultiHeadAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        
        # Feedforward network using SwiGLU
        self.ffn = SWIGLU(d_model, d_ff)

        # Layer norms (RMSNorms) before attention and FFN (Pre-Norm architecture)
        self.rms_rope = RMSNorm(d_model)
        self.rms_ffn = RMSNorm(d_model)

    def forward(self, X):
        """
        Args:
            X (Tensor): Input of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Output of shape (batch_size, seq_len, d_model)
        """
        residual = X
        X = self.rms_rope(X)
        X = self.rope_attention(X)
        X += residual  # Residual connection

        residual = X
        X = self.rms_ffn(X)
        X = self.ffn(X)
        return X + residual  # Second residual connection


class TransformerLM(nn.Module):
    """
    A full Transformer-based language model with:
      - Token embedding layer
      - Stack of transformer blocks
      - Final normalization
      - Linear output projection to vocab size
    """

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()

        # Token embedding layer: maps token IDs to d_model-dim vectors
        self.token_emb = Embedding(vocab_size, d_model, device, dtype)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])

        # Final normalization before output head
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Output projection: from d_model to vocab size
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, X):
        """
        Args:
            X (Tensor): Input token IDs of shape (batch_size, seq_len)
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, vocab_size)
        """
        X = self.token_emb(X)

        for block in self.blocks:
            X = block(X)

        X = self.norm(X)
        logits = self.lm_head(X)
        return logits