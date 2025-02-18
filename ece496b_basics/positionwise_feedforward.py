import math
import torch
import torch.nn as nn

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function.

    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network as described in the Transformer architecture.
    
    Computes:
    
         FFN(x) = GELU(x W1) W2,
    
    where:
      - W1 maps from d_model to d_ff,
      - W2 maps from d_ff back to d_model.
    
    Biases are omitted in both linear transformations.
    """
    def __init__(self, d_model: int, d_ff: int = None):
        """
        Args:
            d_model: Input (and output) dimensionality.
            d_ff: Inner layer dimensionality. Defaults to 4*d_model if not provided.
        """
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).
        """
        return self.w2(gelu(self.w1(x)))