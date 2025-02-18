import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Given an input tensor x with shape (..., d_model), RMSNorm rescales
    each element as follows:
    
        RMSNorm(x)_i = (x_i / RMS(x)) * weight_i,
        
    where:
        RMS(x) = sqrt(mean(x^2) + eps)
        
    weight_i is a learnable gain parameter (of shape [d_model]) typically initialized to 1,
    and eps is a small constant to avoid division by zero.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Renamed parameter to "weight" to match the test fixture.
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the RMS value across the last dimension.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize x and rescale using the weight parameter.
        return (x / rms) * self.weight