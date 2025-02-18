import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your building blocks.
from ece496b_basics.rmsnorm import RMSNorm
from ece496b_basics.multihead_self_attention import MultiHeadSelfAttention
from ece496b_basics.positionwise_feedforward import PositionwiseFeedForward
from ece496b_basics.transformer_block import TransformerBlock  # assuming this file contains the TransformerBlock class

class TransformerLM(nn.Module):
    """
    Transformer Language Model.
    
    This model:
      1. Converts token indices to embeddings and adds learned positional embeddings.
      2. Applies dropout to the sum of these embeddings.
      3. Passes the result through a stack of Transformer blocks.
      4. Applies a final RMSNorm.
      5. Projects the normalized representations to vocabulary logits.
      
    Args:
        vocab_size (int): Size of the vocabulary.
        context_length (int): Maximum context length (for positional embeddings).
        d_model (int): Dimensionality of token embeddings and hidden states.
        num_layers (int): Number of Transformer blocks.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the inner feed-forward layer.
        attn_pdrop (float): Dropout rate for attention probabilities.
        residual_pdrop (float): Dropout rate applied after sublayers and embeddings.
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        
        # Token and positional embeddings.
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.emb_dropout = nn.Dropout(residual_pdrop)
        
        # Stack of Transformer blocks.
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization.
        self.ln_final = RMSNorm(d_model)
        
        # Output projection: maps from d_model to vocab_size.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            tokens: Tensor of shape (batch_size, seq_len) containing token indices.
        
        Returns:
            Tensor of shape (batch_size, seq_len, vocab_size) with unnormalized logits.
        """
        B, T = tokens.size()
        device = tokens.device

        # Lookup token embeddings.
        token_emb = self.token_embeddings(tokens)  # (B, T, d_model)
        
        # Create positional embeddings for positions 0, 1, ..., T-1.
        positions = torch.arange(T, device=device)
        pos_emb = self.position_embeddings(positions)  # (T, d_model)
        
        # Combine token and position embeddings.
        x = token_emb + pos_emb.unsqueeze(0)  # (B, T, d_model)
        x = self.emb_dropout(x)
        
        # Pass through the stack of Transformer blocks.
        for block in self.layers:
            x = block(x)
        
        # Final normalization.
        x = self.ln_final(x)
        
        # Project to vocabulary logits.
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits