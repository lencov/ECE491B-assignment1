import math
import torch
import torch.nn.functional as F

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax.
    
    Subtracts the max over the specified dimension for numerical stability.
    
    Args:
        x: Input tensor.
        dim: Dimension along which to apply softmax.
    
    Returns:
        Tensor of the same shape as x where the entries along `dim` form a probability distribution.
    """
    # Subtract the max value along the given dimension for numerical stability.
    x_max, _ = x.max(dim=dim, keepdim=True)
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.BoolTensor = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Args:
        query: Tensor of shape (batch_size, ..., seq_len_q, d_k)
        key:   Tensor of shape (batch_size, ..., seq_len_k, d_k)
        value: Tensor of shape (batch_size, ..., seq_len_k, d_v)
        mask:  Optional boolean tensor of shape (seq_len_q, seq_len_k) (or broadcastable to that shape).
               Positions where mask is True will be ignored (set to -∞ before softmax).
        dropout_p: Dropout probability to apply on the attention scores.
        training: Whether the model is in training mode (affects dropout).
    
    Returns:
        A tensor of shape (batch_size, ..., seq_len_q, d_v) representing the attended output.
    """
    d_k = query.size(-1)
    scale = math.sqrt(d_k)
    
    # Compute scaled dot-product scores.
    # key.transpose(-2, -1) swaps the last two dims so that we can do matmul.
    scores = torch.matmul(query, key.transpose(-2, -1)) / scale
    
    # If a mask is provided, set masked positions to -infinity.
    if mask is not None:
        scores = scores.masked_fill(mask, -float('inf'))
    
    # Compute softmax over the last dimension (keys).
    attn_probs = softmax(scores, dim=-1)
    
    # Optionally apply dropout on the attention probabilities.
    if dropout_p > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout_p, training=training)
    
    # Multiply the probabilities by the values.
    output = torch.matmul(attn_probs, value)
    return output