import regex as re
import collections
from pathlib import Path

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
):
    """
    1) Read text
    2) Pre-tokenize using regex
    3) Convert each pre-token to UTF-8 bytes and store frequencies
    4) Iteratively find the most frequent adjacent pair of bytes in each 
       token and merge them until we reach vocab_size or no merges remain

    Returns:
      - vocab: dict[int, bytes] - mapping from int (token ID) to token bytes.
      - merges: list[tuple[bytes, bytes]] - list of BPE merges in order.
    """
    if special_tokens is None:
        special_tokens = []
    
    path = Path(input_path)
    text = path.read_text(encoding="utf-8")

    # Pre-tokenization using GPT-2 regex (compiled once)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = re.findall(PAT, text)

    # Convert pre-tokens to bytes and count frequencies
    freq_dict = collections.Counter(pt.encode("utf-8", errors="strict") for pt in pre_tokens)

    # Map each pre-token (as a tuple of single-byte bytes) to its frequency
    symbol_seq_freq = {}
    for bstring, count in freq_dict.items():
        # Instead of building a new tuple for each character repeatedly, we use a generator expression wrapped in tuple.
        symbol_tuple = tuple(bytes((ch,)) for ch in bstring)
        symbol_seq_freq[symbol_tuple] = symbol_seq_freq.get(symbol_tuple, 0) + count

    merges_list = []
    # Starting vocab: 256 single-byte tokens + special tokens
    current_vocab_size = 256 + len(special_tokens)
    max_merges = max(0, vocab_size - current_vocab_size)

    # Main merge loop
    for _ in range(max_merges):
        pair_counts = collections.Counter()
        # Local variable assignment for speed
        for seq, f in symbol_seq_freq.items():
            seq_len = len(seq)
            if seq_len < 2:
                continue
            # Use local variable for pair_counts.update
            for i in range(seq_len - 1):
                pair = (seq[i], seq[i+1])
                pair_counts[pair] += f

        if not pair_counts:
            break

        # Find the best (most frequent) pair; ties broken lexicographically.
        best_pair, best_pair_freq = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        if best_pair_freq == 0:
            break

        merges_list.append(best_pair)
        new_symbol_seq_freq = {}
        # Merge best_pair in all sequences
        for seq, f in symbol_seq_freq.items():
            seq_len = len(seq)
            merged_seq = []
            i = 0
            while i < seq_len:
                # Check if we can merge at this position
                if i < seq_len - 1 and seq[i] == best_pair[0] and seq[i+1] == best_pair[1]:
                    # Merge by concatenating the two bytes
                    merged_seq.append(seq[i] + seq[i+1])
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            # Convert list to tuple once for the dictionary key
            t_merged = tuple(merged_seq)
            new_symbol_seq_freq[t_merged] = new_symbol_seq_freq.get(t_merged, 0) + f

        symbol_seq_freq = new_symbol_seq_freq
        current_vocab_size += 1
        if current_vocab_size >= vocab_size:
            break

    # Build final vocabulary (do this only once at the end)
    vocab = {}
    idx = 0
    # a) Special tokens
    for sp in special_tokens:
        vocab[idx] = sp.encode("utf-8")
        idx += 1
    # b) 256 single-byte tokens
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1
    # c) Merged tokens (in the order applied)
    for pair in merges_list:
        vocab[idx] = pair[0] + pair[1]
        idx += 1

    return vocab, merges_list