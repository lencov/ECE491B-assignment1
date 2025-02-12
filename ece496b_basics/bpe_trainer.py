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
    2) Pre-toxenize using regex
    3) Convert each pre-token to UTF-8 bytes and store frequencies
    4) Iteratively find the most frequent adjacent pair of bytes in each 
        token and merge them until we reach vocab_size or no merges remain
    
    Return:
    - vocab: dict[int, bytes] - mapping from int (token ID in the vocabulary) to bytes (token bytes).
    - merges: list[tuple[bytes, bytes]] - A list of BPE merges produced from training
    """

    if special_tokens is None:
        special_tokens = []
    
    path = Path(input_path)
    text = path.read_text(encoding="utf-8")

    # Pre-tokenization with GPT-2 tokenizer regex
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = re.findall(PAT, text)

    # Convert pre-tokens to bytes and gather frequencies
    freq_dict = collections.Counter()
    for pt in pre_tokens:
        pt_bytes = pt.encode("utf-8", errors="strict")
        freq_dict[pt_bytes] += 1

    # store the sequences in a dict mapping from a tuple of bytes to the frequency
    symbol_seq_freq = {}
    for bstring, count in freq_dict.items():
        symbol_tuple = tuple(bytes([ch]) for ch in bstring)
        symbol_seq_freq[symbol_tuple] = symbol_seq_freq.get(symbol_tuple, 0) + count

    merges_list = []
    current_vocab_size = 256 + len(special_tokens)
    max_merges = vocab_size - current_vocab_size

    for _ in range(max_merges):
        # count all adjacent pairs
        pair_counts = collections.Counter()
        for seq, f in symbol_seq_freq.items():
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += f
        
        if not pair_counts:
            break

        # find the most common pair
        best_pair, best_pair_freq = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        if best_pair_freq == 0:
            break

        # merge best_pair in all sequences
        # Merge best_pair in all sequences
        merges_list.append(best_pair)  # Track the applied merge
        new_symbol_seq_freq = {}

        for seq, f in symbol_seq_freq.items():
            merged_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq)-1 and (seq[i], seq[i+1]) == best_pair:
                    # Merge the best pair
                    merged_seq.append(seq[i] + seq[i+1])  # e.g. b"e" + b"s" = b"es"
                    i += 2  # Skip next token since it's merged
                else:
                    merged_seq.append(seq[i])
                    i += 1
            # Update frequency counts for the new merged sequence
            new_symbol_seq_freq[tuple(merged_seq)] = new_symbol_seq_freq.get(tuple(merged_seq), 0) + f
    
        symbol_seq_freq = new_symbol_seq_freq
        current_vocab_size += 1
        if current_vocab_size >= vocab_size:
            break

        # 5) Build final vocab
        vocab = {}
        idx = 0
        # a) Add special tokens
        for sp in special_tokens:
            vocab[idx] = sp.encode("utf-8")
            idx += 1
        # b) Add 256 single bytes
        for b in range(256):
            vocab[idx] = bytes([b])
            idx += 1
        # c) Add merges in order
        for pair in merges_list:
            vocab[idx] = pair[0] + pair[1]
            idx += 1

        return vocab, merges_list