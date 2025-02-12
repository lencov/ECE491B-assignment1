import regex as re
import collections
import multiprocessing
from pathlib import Path

# Move count_chunk to module level so it can be pickled.
def count_chunk(chunk):
    local_counts = collections.Counter()
    for seq, f in chunk:
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            local_counts[pair] += f
    return local_counts

def count_pairs_parallel(symbol_seq_freq):
    pair_counts = collections.Counter()
    # Use at most 2 workers.
    num_workers = min(2, multiprocessing.cpu_count())
    # Compute chunk size; avoid zero-length chunks.
    chunk_size = len(symbol_seq_freq) // num_workers or len(symbol_seq_freq)
    items = list(symbol_seq_freq.items())
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(count_chunk, chunks)
    
    for res in results:
        pair_counts.update(res)
    return pair_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    1) Read text
    2) Pre-tokenize using regex
    3) Convert each pre-token to UTF-8 bytes and store frequencies
    4) Iteratively find the most frequent adjacent pair of bytes in each 
       token and merge them until we reach vocab_size or no merges remain

    Returns:
      - vocab: dict[int, bytes] - mapping from token ID to token bytes.
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
        symbol_tuple = tuple(bytes((ch,)) for ch in bstring)
        symbol_seq_freq[symbol_tuple] = symbol_seq_freq.get(symbol_tuple, 0) + count

    merges_list = []
    # Starting vocab: 256 single-byte tokens + special tokens
    current_vocab_size = 256 + len(special_tokens)
    max_merges = max(0, vocab_size - current_vocab_size)

    # Main merge loop
    for _ in range(max_merges):
        pair_counts = count_pairs_parallel(symbol_seq_freq)
        if not pair_counts:
            break

        best_pair, best_pair_freq = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        if best_pair_freq == 0:
            break

        merges_list.append(best_pair)
        new_symbol_seq_freq = {}
        for seq, f in symbol_seq_freq.items():
            seq_len = len(seq)
            merged_seq = []
            i = 0
            while i < seq_len:
                if i < seq_len - 1 and seq[i] == best_pair[0] and seq[i+1] == best_pair[1]:
                    merged_seq.append(seq[i] + seq[i+1])
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            t_merged = tuple(merged_seq)
            new_symbol_seq_freq[t_merged] = new_symbol_seq_freq.get(t_merged, 0) + f

        symbol_seq_freq = new_symbol_seq_freq
        current_vocab_size += 1
        if current_vocab_size >= vocab_size:
            break

    # Build final vocabulary (only once at the end)
    vocab = {}
    idx = 0
    # a) Add special tokens
    for sp in special_tokens:
        vocab[idx] = sp.encode("utf-8")
        idx += 1
    # b) Add 256 single-byte tokens
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1
    # c) Add merged tokens in order of application
    for pair in merges_list:
        vocab[idx] = pair[0] + pair[1]
        idx += 1

    return vocab, merges_list