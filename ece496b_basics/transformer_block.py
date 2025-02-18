import torch
import torch.nn as nn

from ece496b_basics.rmsnorm import RMSNorm
from ece496b_basics.multihead_self_attention import MultiHeadSelfAttention
from ece496b_basics.positionwise_feedforward import PositionwiseFeedForward

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    This block implements two sublayers:
      1. Multi-head self-attention:
           y = x + Dropout(MultiHeadSelfAttention(RMSNorm(x)))
      2. Feed-forward network:
           z = y + Dropout(PositionwiseFeedForward(RMSNorm(y)))
    
    Args:
        d_model (int): Dimensionality of the Transformer block inputs.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the inner layer in the feed-forward network.
        attn_pdrop (float): Dropout rate for attention probabilities.
        residual_pdrop (float): Dropout rate applied after each sublayer.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float, residual_pdrop: float):
        super().__init__()
        # Sublayer 1: Multi-head self-attention
        self.norm1 = RMSNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        self.dropout1 = nn.Dropout(residual_pdrop) if residual_pdrop > 0.0 else nn.Identity()

        # Sublayer 2: Feed-forward network
        self.norm2 = RMSNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(residual_pdrop) if residual_pdrop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sublayer 1: Multi-head self-attention with pre-norm and residual connection.
        attn_out = self.mhsa(self.norm1(x))
        x = x + self.dropout1(attn_out)
        
        # Sublayer 2: Position-wise feed-forward network with pre-norm and residual connection.
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout2(ff_out)
        return x
