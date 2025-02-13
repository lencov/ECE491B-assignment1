import regex as re
import collections
from pathlib import Path
import time
import logging

# Configure logging to display INFO messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    t0 = time.perf_counter()
    
    if special_tokens is None:
        special_tokens = []
    
    # 1. Read text
    path = Path(input_path)
    text = path.read_text(encoding="utf-8")
    t_read = time.perf_counter()
    logging.info(f"Reading text took: {t_read - t0:.4f} seconds")

    # 2. Pre-tokenization using GPT-2 regex (compiled once)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = re.findall(PAT, text)
    t_pre = time.perf_counter()
    logging.info(f"Pre-tokenization took: {t_pre - t_read:.4f} seconds; found {len(pre_tokens)} tokens")

    # 3. Convert pre-tokens to bytes and gather frequencies
    freq_dict = collections.Counter(pt.encode("utf-8", errors="strict") for pt in pre_tokens)
    t_freq = time.perf_counter()
    logging.info(f"Converting tokens to bytes and counting frequencies took: {t_freq - t_pre:.4f} seconds; {len(freq_dict)} unique tokens")

    # 4. Build initial symbol sequence frequency mapping
    symbol_seq_freq = {}
    for bstring, count in freq_dict.items():
        symbol_tuple = tuple(bytes([ch]) for ch in bstring)
        symbol_seq_freq[symbol_tuple] = symbol_seq_freq.get(symbol_tuple, 0) + count
    t_sym = time.perf_counter()
    logging.info(f"Building symbol sequence frequency mapping took: {t_sym - t_freq:.4f} seconds")

    merges_list = []
    # Starting vocab: 256 single-byte tokens + special tokens
    current_vocab_size = 256 + len(special_tokens)
    max_merges = max(0, vocab_size - current_vocab_size)

    # Main merge loop timing each iteration
    merge_loop_total = 0.0
    num_iters = 0
    for _ in range(max_merges):
        iter_start = time.perf_counter()
        # Count adjacent pairs
        pair_counts = collections.Counter()
        for seq, f in symbol_seq_freq.items():
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += f

        if not pair_counts:
            break

        best_pair, best_pair_freq = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        if best_pair_freq == 0:
            break

        merges_list.append(best_pair)
        new_symbol_seq_freq = {}
        for seq, f in symbol_seq_freq.items():
            merged_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                    merged_seq.append(seq[i] + seq[i + 1])
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            new_symbol_seq_freq[tuple(merged_seq)] = new_symbol_seq_freq.get(tuple(merged_seq), 0) + f

        symbol_seq_freq = new_symbol_seq_freq
        current_vocab_size += 1
        num_iters += 1
        iter_end = time.perf_counter()
        iter_time = iter_end - iter_start
        merge_loop_total += iter_time
        logging.info(f"Iteration {num_iters}: best pair {best_pair} (freq={best_pair_freq}) merged in {iter_time:.4f} seconds")
        if current_vocab_size >= vocab_size:
            break

    logging.info(f"Total merge loop time for {num_iters} iterations: {merge_loop_total:.4f} seconds")

    # Build final vocabulary (only once at the end)
    t_vocab_start = time.perf_counter()
    vocab = {}
    idx = 0
    for sp in special_tokens:
        vocab[idx] = sp.encode("utf-8")
        idx += 1
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1
    for pair in merges_list:
        vocab[idx] = pair[0] + pair[1]
        idx += 1
    t_vocab_end = time.perf_counter()
    logging.info(f"Building final vocabulary took: {t_vocab_end - t_vocab_start:.4f} seconds")

    total_time = time.perf_counter() - t0
    logging.info(f"Total training time: {total_time:.4f} seconds")
    
    return vocab, merges_list