import regex as re
import collections
from pathlib import Path

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    1) Read text
    2) Pre-tokenize using regex
    3) Convert each pre-token to a tuple of ints (each representing a single byte)
    4) Iteratively find the most frequent adjacent pair (of ints) in each 
       token and merge them until we reach vocab_size or no merges remain.
    
    Returns:
        vocab: dict[int, bytes] - mapping from token ID to token bytes.
        merges: list[tuple[bytes, bytes]] - list of BPE merges (as bytes pairs)
    """
    if special_tokens is None:
        special_tokens = []
    
    path = Path(input_path)
    text = path.read_text(encoding="utf-8")
    
    # Pre-tokenization using GPT-2 regex
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = re.findall(PAT, text)
    
    # Build a frequency dictionary of words.
    # Instead of storing each word as a tuple of bytes objects, we store it as a tuple of ints.
    symbol_seq_freq = collections.Counter()
    for pt in pre_tokens:
        b = pt.encode("utf-8", errors="strict")
        # Represent the token as a tuple of ints (each int in 0-255)
        symbol_seq = tuple(b)
        symbol_seq_freq[symbol_seq] += 1

    # We'll use an integer representation for tokens.
    # For base tokens (0-255) the integer is the same as the byte value.
    # When we merge, we assign new integer IDs (starting at 256).
    token2bytes = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    merges_list = []  # Will store the merge operations as pairs of integers

    # The current vocabulary already contains 256 base tokens plus the special tokens.
    current_vocab_size = 256 + len(special_tokens)
    max_merges = max(0, vocab_size - current_vocab_size)

    for _ in range(max_merges):
        pair_counts = collections.Counter()
        # Count frequencies of adjacent pairs (as tuples of ints)
        for seq, freq in symbol_seq_freq.items():
            if len(seq) < 2:
                continue
            # zip(seq, seq[1:]) yields adjacent pairs
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += freq

        if not pair_counts:
            break

        # Find the most frequent pair.
        best_pair, best_count = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        if best_count == 0:
            break

        # Record the merge.
        merges_list.append(best_pair)
        new_token = next_token_id
        next_token_id += 1
        # Create new merged token's bytes by concatenating the two parts.
        token2bytes[new_token] = token2bytes[best_pair[0]] + token2bytes[best_pair[1]]

        # Merge the best pair in all sequences.
        new_symbol_seq_freq = {}
        for seq, freq in symbol_seq_freq.items():
            merged_seq = []
            i = 0
            seq_len = len(seq)
            while i < seq_len:
                # If the best pair is found, merge it into new_token.
                if i < seq_len - 1 and (seq[i], seq[i+1]) == best_pair:
                    merged_seq.append(new_token)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            new_symbol_seq = tuple(merged_seq)
            new_symbol_seq_freq[new_symbol_seq] = new_symbol_seq_freq.get(new_symbol_seq, 0) + freq

        symbol_seq_freq = new_symbol_seq_freq
        current_vocab_size += 1
        if current_vocab_size >= vocab_size:
            break

    # Build the final vocabulary.
    vocab = {}
    idx = 0

    for sp in special_tokens:
        vocab[idx] = sp.encode("utf-8")
        idx += 1
    #  Base tokens
    for i in range(256):
        vocab[idx] = token2bytes[i]
        idx += 1
    # Merge tokens
    for merge_index in range(len(merges_list)):
        token_id = 256 + merge_index
        vocab[idx] = token2bytes[token_id]
        idx += 1

    merges_bytes = [(token2bytes[a], token2bytes[b]) for (a, b) in merges_list]
    return vocab, merges_bytes