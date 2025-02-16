import math
import torch
import torch.nn as nn

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function.

    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

    Args:
        x: Input tensor.

    Returns:
        Tensor with GELU activation applied.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network as described in the Transformer architecture.
    
    This module applies two linear transformations with a GELU activation in between,
    and omits biases in both linear transformations.
    
    Given an input x with shape (batch_size, seq_len, d_model), the module computes:
    
         FFN(x) = GELU(x W1) W2,
    
    where:
      - W1 is a linear transformation mapping from d_model to d_ff (with d_ff = 4 * d_model),
      - W2 is a linear transformation mapping from d_ff back to d_model.
    """
    def __init__(self, d_model: int):
        """
        Args:
            d_model: The input (and output) dimensionality of the model.
        """
        super().__init__()
        d_ff = 4 * d_model
        # Linear transformation from d_model to d_ff without bias.
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        # Linear transformation from d_ff back to d_model without bias.
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the position-wise feed-forward network.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).
        """
        return self.fc2(gelu(self.fc1(x)))