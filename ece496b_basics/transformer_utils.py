import math
import torch
import torch.nn.functional as F
from collections.abc import Iterable
import numpy as np
from typing import Union, IO

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


def remap_transformer_block_state_dict(state_dict: dict, num_heads: int, d_k: int) -> dict:
    """
    Remap keys in the provided state dict to match the naming in our TransformerBlock.
    
    Expected reference keys (from the fixture):
      - "attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight",
        "attn.output_proj.weight",
      - "ln1.weight", "ffn.w1.weight", "ffn.w2.weight", "ln2.weight"
      
    We want to map them to our internal module names:
      - attn.q_heads, attn.k_heads, attn.v_heads, attn.output_proj,
      - ln1, ffn, ln2 remain the same.
    
    This function splits the concatenated projection weights into per-head weights.
    """
    new_state_dict = dict(state_dict)  # copy to avoid mutating input

    # Remap the attention projections.
    if "attn.q_proj.weight" in new_state_dict:
        q_proj = new_state_dict.pop("attn.q_proj.weight")  # shape: (num_heads * d_k, d_model)
        # Split q_proj into a tuple of tensors, each of shape (d_k, d_model)
        q_heads = torch.stack(torch.split(q_proj, d_k, dim=0), dim=0)
        new_state_dict["attn.q_heads"] = q_heads

    if "attn.k_proj.weight" in new_state_dict:
        k_proj = new_state_dict.pop("attn.k_proj.weight")
        k_heads = torch.stack(torch.split(k_proj, d_k, dim=0), dim=0)
        new_state_dict["attn.k_heads"] = k_heads

    if "attn.v_proj.weight" in new_state_dict:
        v_proj = new_state_dict.pop("attn.v_proj.weight")
        v_heads = torch.stack(torch.split(v_proj, d_k, dim=0), dim=0)
        new_state_dict["attn.v_heads"] = v_heads

    if "attn.output_proj.weight" in new_state_dict:
        # Simply rename key; our module expects "attn.output_proj"
        new_state_dict["attn.output_proj"] = new_state_dict.pop("attn.output_proj.weight")

    # The other keys (ln1.weight, ffn.w1.weight, ffn.w2.weight, ln2.weight)
    # already match our naming in TransformerBlock if we use ln1, ffn, and ln2.

    return new_state_dict

def remap_transformer_lm_state_dict(state_dict: dict, num_layers: int, num_heads: int, d_model: int) -> dict:
    """
    Remap keys for the full TransformerLM model.
    
    This function ensures that the state dict keys match the expected structure of TransformerLM.
    """
    new_state_dict = {}

    # Remap embeddings
    new_state_dict["token_embeddings.weight"] = state_dict["token_embeddings.weight"]
    new_state_dict["position_embeddings.weight"] = state_dict["position_embeddings.weight"]
    new_state_dict["ln_final.weight"] = state_dict["ln_final.weight"]
    new_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]

    # Remap each Transformer block
    d_k = d_model // num_heads
    for layer_idx in range(num_layers):
        prefix_old = f"layers.{layer_idx}."
        prefix_new = f"layers.{layer_idx}."

        # Remap attention projections
        new_state_dict[f"{prefix_new}attn.q_heads"] = torch.stack(
            torch.split(state_dict[f"{prefix_old}attn.q_proj.weight"], d_k, dim=0), dim=0
        )
        new_state_dict[f"{prefix_new}attn.k_heads"] = torch.stack(
            torch.split(state_dict[f"{prefix_old}attn.k_proj.weight"], d_k, dim=0), dim=0
        )
        new_state_dict[f"{prefix_new}attn.v_heads"] = torch.stack(
            torch.split(state_dict[f"{prefix_old}attn.v_proj.weight"], d_k, dim=0), dim=0
        )
        new_state_dict[f"{prefix_new}attn.output_proj"] = state_dict[f"{prefix_old}attn.output_proj.weight"]

        # Copy layer norms and feed-forward weights
        new_state_dict[f"{prefix_new}ln1.weight"] = state_dict[f"{prefix_old}ln1.weight"]
        new_state_dict[f"{prefix_new}ffn.w1.weight"] = state_dict[f"{prefix_old}ffn.w1.weight"]
        new_state_dict[f"{prefix_new}ffn.w2.weight"] = state_dict[f"{prefix_old}ffn.w2.weight"]
        new_state_dict[f"{prefix_new}ln2.weight"] = state_dict[f"{prefix_old}ln2.weight"]

    return new_state_dict

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the average cross-entropy loss given predicted logits and target indices.
    
    Args:
        logits: Tensor of shape (..., vocab_size) containing the unnormalized logit scores.
        targets: Tensor of shape (...) containing the target indices.
        
    Returns:
        A scalar tensor representing the average cross-entropy loss over all tokens.
    """
    # For numerical stability, subtract the max logit from each vector.
    logits_stable = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # Compute the negative log-likelihood for the correct class.
    # targets.unsqueeze(-1) reshapes targets so that gather works along the vocab dimension.
    loss = -logits_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # Return the average loss over all batch (and any additional) dimensions.
    return loss.mean()


def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int) -> float:
    """
    Returns the learning rate at step t according to a cosine annealing schedule with warmup.

    Args:
        t (int): Current iteration.
        alpha_max (float): Maximum (initial) learning rate.
        alpha_min (float): Minimum (final) learning rate.
        Tw (int): Number of warm-up iterations.
        Tc (int): Total number of iterations for cosine annealing (after warm-up).

    Returns:
        float: The learning rate for iteration t.
    """
    if t < Tw:
        # Warmup: linear increase from 0 to alpha_max.
        return (t / Tw) * alpha_max
    elif t <= Tc:
        # Cosine annealing between Tw and Tc.
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (t - Tw) / (Tc - Tw)))
        return alpha_min + cosine_decay * (alpha_max - alpha_min)
    else:
        # After Tc, the learning rate is fixed at alpha_min.
        return alpha_min
    
def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Given a set of parameters, clip their combined gradients so that their total L2 norm
    is at most max_l2_norm. The gradients are modified in-place.

    Args:
        parameters: Iterable of trainable parameters.
        max_l2_norm: A positive value specifying the maximum allowed L2 norm.
        
    Returns:
        None
    """
    eps = 1e-6  # numerical stability
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm_sq += p.grad.data.pow(2).sum().item()
    total_norm = total_norm_sq ** 0.5

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    Samples a batch of training sequences from a concatenated sequence of token IDs.

    Args:
      x (np.ndarray): A 1D numpy array containing token IDs.
      batch_size (int): Number of sequences in the batch.
      context_length (int): Length of each input sequence (and corresponding target).
      device (str): PyTorch device (e.g., 'cpu' or 'cuda:0').

    Returns:
      A tuple (inputs, targets) where:
         - inputs is a torch.Tensor of shape (batch_size, context_length)
         - targets is a torch.Tensor of shape (batch_size, context_length)
      Both tensors are placed on the specified device.
      
    Note:
      For each sampled starting index i (with 0 ≤ i ≤ n - context_length - 1),
      the input sequence is x[i : i+context_length] and the target is x[i+1 : i+1+context_length].
    """
    n = x.shape[0]
    # Valid starting indices are 0 to n - context_length - 1 inclusive,
    # so we set high = n - context_length to sample from [0, n - context_length)
    max_start = n - context_length
    indices = np.random.randint(0, max_start, size=batch_size)
    
    # Build the input and target arrays.
    inputs = np.stack([x[i:i+context_length] for i in indices])
    targets = np.stack([x[i+1:i+1+context_length] for i in indices])
    
    # Convert the arrays to PyTorch tensors and move them to the specified device.
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    
    return inputs, targets

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: Union[str, bytes, IO[bytes]]) -> None:
    """
    Saves a checkpoint containing the model and optimizer states as well as the current iteration.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        iteration: The current iteration number.
        out: The destination path or file-like object to save the checkpoint.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: Union[str, bytes, IO[bytes]],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    """
    Loads a checkpoint from the given source and restores the model and optimizer states.

    Args:
        src: The source path or file-like object to load the checkpoint.
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.

    Returns:
        The iteration number saved in the checkpoint.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]