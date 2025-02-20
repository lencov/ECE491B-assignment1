import torch
import torch.nn.functional as F

def top_p_filtering(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """
    Filters logits using nucleus (top-p) sampling.
    
    Args:
        logits: 1D tensor of logits (shape: vocab_size).
        top_p: Cumulative probability threshold.
    
    Returns:
        Filtered logits tensor with values below the top-p threshold set to -inf.
    """
    # Sort logits descending and compute softmax probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Determine which indices to remove: those that push cumulative prob over top_p.
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the mask one token to the right to include the token that crosses the threshold.
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0  # always keep the top token

    # Set logits of tokens to be removed to -infinity.
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')
    return logits

def generate_text(
    model: torch.nn.Module,
    prompt: list[int] | torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    context_length: int = 128,
    end_token_id: int = None,
    device: str = 'cuda:0'
) -> list[int]:
    """
    Generates text from a language model.
    
    Args:
        model: The trained language model.
        prompt: A list of token IDs or a 1D tensor of token IDs serving as the prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Temperature for softmax scaling.
        top_p: Nucleus (top-p) sampling threshold.
        context_length: Maximum context length to feed into the model.
        end_token_id: Optional token ID that, when generated, terminates generation.
        device: PyTorch device to run the model on.
        
    Returns:
        A list of token IDs representing the generated sequence (including the prompt).
    """
    model.eval()
    
    # Ensure prompt is a 2D tensor with shape (1, prompt_length).
    if not torch.is_tensor(prompt):
        prompt = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    elif prompt.dim() == 1:
        prompt = prompt.unsqueeze(0).to(device)
    else:
        prompt = prompt.to(device)
    
    generated = prompt
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Use only the last context_length tokens.
            input_seq = generated[:, -context_length:] if generated.size(1) > context_length else generated
            # Get logits from the model; shape: (1, T, vocab_size).
            logits = model(input_seq)
            # Take the logits corresponding to the last time step.
            next_logits = logits[0, -1, :]
            # Apply temperature scaling.
            next_logits = next_logits / temperature
            # Apply top-p (nucleus) filtering.
            filtered_logits = top_p_filtering(next_logits.clone(), top_p=top_p)
            # Convert logits to probabilities.
            probs = F.softmax(filtered_logits, dim=-1)
            # Sample the next token.
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # shape: (1, 1)
            # Append to the generated sequence.
            generated = torch.cat((generated, next_token), dim=1)
            
            # Stop if the end-of-text token is generated.
            if end_token_id is not None and next_token.item() == end_token_id:
                break

    # Return the sequence as a list of token IDs.
    return generated.squeeze(0).tolist()