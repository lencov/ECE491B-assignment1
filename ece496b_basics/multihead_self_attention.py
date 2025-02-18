import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention.
    
    Given an input tensor x of shape (batch_size, seq_len, d_model),
    this module computes multi-head self-attention as in Vaswani et al. (2017).
    
    The projection matrices for Q, K, V are stored as parameters of shape 
    (num_heads, d_k, d_model) where d_k = d_model // num_heads.
    The output projection is a parameter of shape (d_model, num_heads * d_v),
    with d_v = d_k.
    
    Causal masking is applied so that token i cannot attend to tokens j > i.
    """
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float):
        """
        Args:
            d_model: Dimensionality of the model.
            num_heads: Number of attention heads.
            attn_pdrop: Dropout rate on attention probabilities.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # also d_v
        self.d_v = self.d_k

        # Instead of having h separate Linear layers, we combine the weights for all heads.
        # Expected state dict keys (from the reference) are:
        #   "q_heads.{N}.weight" with shape (d_k, d_model)
        #   "k_heads.{N}.weight" with shape (d_k, d_model)
        #   "v_heads.{N}.weight" with shape (d_v, d_model)
        # We will register a single parameter for each projection of shape
        # (num_heads, d_k, d_model) (or (num_heads, d_v, d_model) for V).
        self.q_heads = nn.Parameter(torch.empty(num_heads, self.d_k, d_model))
        self.k_heads = nn.Parameter(torch.empty(num_heads, self.d_k, d_model))
        self.v_heads = nn.Parameter(torch.empty(num_heads, self.d_v, d_model))
        # Output projection weight: shape (d_model, num_heads * d_v)
        self.output_proj = nn.Parameter(torch.empty(d_model, num_heads * self.d_v))
        
        self.attn_pdrop = attn_pdrop
        self.dropout = nn.Dropout(attn_pdrop) if attn_pdrop > 0.0 else nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize weights with Xavier uniform.
        nn.init.xavier_uniform_(self.q_heads)
        nn.init.xavier_uniform_(self.k_heads)
        nn.init.xavier_uniform_(self.v_heads)
        nn.init.xavier_uniform_(self.output_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) after applying multi-head self-attention.
        """
        batch_size, seq_len, _ = x.shape

        # --- Compute Q, K, V in one go ---
        # First, combine the per-head weights into a 2D weight for each projection.
        # q_weight: shape (num_heads*d_k, d_model)
        q_weight = self.q_heads.view(self.num_heads * self.d_k, self.d_model)
        k_weight = self.k_heads.view(self.num_heads * self.d_k, self.d_model)
        v_weight = self.v_heads.view(self.num_heads * self.d_v, self.d_model)
        # Compute projections: result shapes (batch_size, seq_len, num_heads*d_k)
        Q = torch.matmul(x, q_weight.t())
        K = torch.matmul(x, k_weight.t())
        V = torch.matmul(x, v_weight.t())
        
        # --- Reshape to separate heads ---
        # New shape: (batch_size, seq_len, num_heads, d_k) then transpose to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # --- Scaled Dot-Product Attention ---
        # Compute scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Create a causal mask (upper-triangular, excluding diagonal)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        # Softmax over the last dimension
        attn = torch.softmax(scores, dim=-1)
        # Apply dropout on attention probabilities
        attn = self.dropout(attn)
        # Compute weighted sum of values: (batch_size, num_heads, seq_len, d_v)
        out = torch.matmul(attn, V)
        
        # --- Concatenate heads and project ---
        # Transpose and reshape: (batch_size, seq_len, num_heads*d_v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_v)
        # Final output projection: (batch_size, seq_len, d_model)
        out = torch.matmul(out, self.output_proj.t())
        return out

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict to combine per-head weights from the state dict into our parameters.
        The reference state dict provides keys of the form:
          "q_heads.{N}.weight", "k_heads.{N}.weight", "v_heads.{N}.weight"
        for each head index N. We pop these and stack them to form our combined parameters.
        """
        q_list, k_list, v_list = [], [], []
        for n in range(self.num_heads):
            q_key = f"q_heads.{n}.weight"
            k_key = f"k_heads.{n}.weight"
            v_key = f"v_heads.{n}.weight"
            if q_key not in state_dict or k_key not in state_dict or v_key not in state_dict:
                raise KeyError(f"Missing projection key for head {n} in state_dict")
            q_list.append(state_dict.pop(q_key))
            k_list.append(state_dict.pop(k_key))
            v_list.append(state_dict.pop(v_key))
        # Stack to form tensors of shape (num_heads, d_k, d_model)
        state_dict["q_heads"] = torch.stack(q_list, dim=0)
        state_dict["k_heads"] = torch.stack(k_list, dim=0)
        state_dict["v_heads"] = torch.stack(v_list, dim=0)
        
        # Rename output projection key if necessary.
        if "output_proj.weight" in state_dict:
            state_dict["output_proj"] = state_dict.pop("output_proj.weight")
        
        return super().load_state_dict(state_dict, strict=strict)
