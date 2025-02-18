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
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        self.dropout1 = nn.Dropout(residual_pdrop) if residual_pdrop > 0.0 else nn.Identity()

        self.ln2 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(residual_pdrop) if residual_pdrop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sublayer 1: Apply pre-norm, multi-head self-attention, dropout, then add residual.
        attn_out = self.attn(self.ln1(x))
        x = x + self.dropout1(attn_out)
        # Sublayer 2: Apply pre-norm, feed-forward network, dropout, then add residual.
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)
        return x
