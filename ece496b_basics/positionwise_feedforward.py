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
    
    Given an input x with shape (..., d_model), this module computes:
    
         FFN(x) = GELU(x W1) W2,
    
    where:
      - W1 is a linear transformation mapping from d_model to d_ff (with d_ff = 4 * d_model),
      - W2 is a linear transformation mapping from d_ff back to d_model.
    
    Note:
      Biases are omitted in both linear transformations.
    """
    def __init__(self, d_model: int):
        """
        Args:
            d_model: The input (and output) dimensionality of the model.
        """
        super().__init__()
        d_ff = 4 * d_model  # as specified in the assignment description
        # Rename the layers to w1 and w2 to match the state dict keys
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (..., d_model).
        
        Returns:
            Output tensor of shape (..., d_model).
        """
        return self.w2(gelu(self.w1(x)))