import regex as re
import collections
from pathlib import Path
import time
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def findall_with_timeout(pattern, chunk, timeout=5):
    """
    Runs pattern.findall(chunk) in a separate thread and waits for up to 'timeout' seconds.
    Returns None if the operation times out.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pattern.findall, chunk)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return None

def stream_tokens(text, pattern, batch_size=10000, timeout=5):
    """
    Generator that yields tokens from text in batches.
    Logs processing for each batch.
    If a batch times out, it logs and skips that batch.
    """
    text_len = len(text)
    for start in range(0, text_len, batch_size):
        end = min(start + batch_size, text_len)
        chunk = text[start:end]
        print(f"Processing batch from {start} to {end} (size {end - start})")
        batch_start_time = time.perf_counter()
        chunk_tokens = findall_with_timeout(pattern, chunk, timeout=timeout)
        batch_end_time = time.perf_counter()
        batch_time = batch_end_time - batch_start_time
        
        if chunk_tokens is None:
            sample = chunk[:100]
            print(f"⚠️ Timeout processing chunk starting at {start} (batch size {batch_size}). Skipping this chunk. Sample: {sample!r}", file=sys.stderr)
            continue
        print(f"  Batch starting at {start} produced {len(chunk_tokens)} tokens in {batch_time:.2f}s")
        for token in chunk_tokens:
            yield token

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    1) Read text from file.
    2) Pre-tokenize using a regex in batches (streaming tokens).
    3) Convert each token to UTF-8 bytes and update frequency counts.
    4) Iteratively merge tokens via BPE until the vocabulary size is reached.

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
    print(f"Reading text took: {t_read - t0:.4f} seconds")
    print(f"File size (in characters): {len(text)}")
    print(f"File sample (first 200 chars): {text[:200]!r}")
    
    # 2. Pre-tokenization using GPT-2 regex (compiled once)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)
    # Stream tokens rather than accumulating them in a list
    token_generator = stream_tokens(text, pattern, batch_size=10000, timeout=5)
    
    # 3. Update frequency counter directly from the generator (streaming)
    freq_dict = collections.Counter(token.encode("utf-8", errors="strict") for token in token_generator)
    t_freq = time.perf_counter()
    print(f"Streaming token frequency counting took: {t_freq - t_read:.4f} seconds; {len(freq_dict)} unique tokens")
    
    # 4. Build initial symbol sequence frequency mapping
    symbol_seq_freq = {}
    for bstring, count in freq_dict.items():
        symbol_tuple = tuple(bytes([ch]) for ch in bstring)
        symbol_seq_freq[symbol_tuple] = symbol_seq_freq.get(symbol_tuple, 0) + count
    t_sym = time.perf_counter()
    print(f"Building symbol sequence frequency mapping took: {t_sym - t_freq:.4f} seconds")
    
    merges_list = []
    # Starting vocab: 256 single-byte tokens + special tokens
    current_vocab_size = 256 + len(special_tokens)
    max_merges = max(0, vocab_size - current_vocab_size)
    
    merge_loop_total = 0.0
    num_iters = 0
    for _ in range(max_merges):
        iter_start = time.perf_counter()
        # Count adjacent pairs in all sequences
        pair_counts = collections.Counter()
        for seq, f in symbol_seq_freq.items():
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
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
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == best_pair:
                    merged_seq.append(seq[i] + seq[i+1])
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
        print(f"Iteration {num_iters}: best pair {best_pair} (freq={best_pair_freq}) merged in {iter_time:.4f} seconds")
        if current_vocab_size >= vocab_size:
            break

    print(f"Total merge loop time for {num_iters} iterations: {merge_loop_total:.4f} seconds")
    
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
    print(f"Building final vocabulary took: {t_vocab_end - t_vocab_start:.4f} seconds")
    
    total_time = time.perf_counter() - t0
    print(f"Total training time: {total_time:.4f} seconds")
    
    return vocab, merges_list