import regex as re
import collections
from pathlib import Path
import time
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# --- Byte encoder: Map byte values (0-255) to Unicode characters.
def bytes_to_unicode():
    """
    Returns a dictionary mapping byte values (0-255) to Unicode characters.
    This is essentially the mapping used in GPT-2's tokenizer.
    """
    bs = list(range(32, 127)) + \
     list(range(161, 173)) + \
     list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

# Get our mapping once.
BYTE_ENCODER = bytes_to_unicode()

def convert_token(token_tuple, byte_encoder=BYTE_ENCODER):
    """
    Converts a token (a tuple of ints) into a UTF-8 bytes object.
    Each integer is mapped through the byte_encoder.
    """
    try:
        s = "".join(byte_encoder[b] for b in token_tuple)
    except TypeError:
        print(f"Warning: Expected token_tuple to be iterable, got: {token_tuple}", file=sys.stderr)
        s = ""
    return s.encode("utf-8")

# --- Token streaming and timeout
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

def stream_tokens(text, pattern, batch_size=100000, timeout=5):
    """
    Generator that yields tokens from text in batches.
    For each token string found by the regex, we split it into a sequence of base tokens.
    Each base token is represented as a one-element tuple of ints.
    
    For example, if the regex finds "Hello", we yield:
       ((72,), (101,), (108,), (108,), (111,))
       
    If a batch times out, it is logged and skipped.
    """
    text_len = len(text)
    for start in range(0, text_len, batch_size):
        end = min(start + batch_size, text_len)
        chunk = text[start:end]
        #print(f"Processing batch from {start} to {end} (size {end - start})")
        batch_start_time = time.perf_counter()
        chunk_tokens = findall_with_timeout(pattern, chunk, timeout=timeout)
        batch_end_time = time.perf_counter()
        batch_time = batch_end_time - batch_start_time
        if chunk_tokens is None:
            sample = chunk[:100]
            print(f"⚠️ Timeout processing chunk starting at {start} (batch size {batch_size}). Skipping this chunk. Sample: {sample!r}", file=sys.stderr)
            continue
        #print(f"  Batch starting at {start} produced {len(chunk_tokens)} tokens in {batch_time:.2f}s")
        for token in chunk_tokens:
            # First, encode the token to bytes.
            token_bytes = token.encode("utf-8", errors="strict")
            # Then, split the bytes into individual base tokens: each as a one-element tuple.
            yield tuple(bytes([b]) for b in token_bytes)

# --- BPE Training
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    1) Read text from file.
    2) Pre-tokenize the text in batches (streaming tokens).
    3) Build a frequency counter directly from the token generator.
    4) Iteratively merge tokens via BPE until the vocabulary size is reached.
    5) Build the final vocabulary:
         - Special tokens (converted to UTF-8 bytes),
         - Base tokens: each byte is mapped via our custom byte encoder,
         - Merged tokens: each token (a tuple of ints) is converted to bytes via convert_token.
         
    Returns:
      - vocab: dict[int, bytes] - mapping from token ID to token bytes.
      - merges: list[tuple[bytes, bytes]] - list of BPE merges (each merge is a tuple of two bytes objects).
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
    token_generator = stream_tokens(text, pattern, batch_size=10000, timeout=5)
    
    # 3. Build frequency counter from the token generator.
    # Each token is now a sequence of base tokens: a tuple of tuples.
    freq_dict = collections.Counter(token for token in token_generator)
    t_freq = time.perf_counter()
    print(f"Streaming token frequency counting took: {t_freq - t_read:.4f} seconds; {len(freq_dict)} unique tokens")
    
    # 4. The initial symbol sequence frequency mapping is just our frequency counter.
    # Each key is a token sequence (a tuple of base tokens), where each base token is a one-element tuple.
    symbol_seq_freq = dict(freq_dict)
    t_sym = time.perf_counter()
    print(f"Building symbol sequence frequency mapping took: {t_sym - t_freq:.4f} seconds")
    
    merges_list = []
    # Starting vocabulary: 256 base tokens + special tokens.
    current_vocab_size = 256 + len(special_tokens)
    max_merges = max(0, vocab_size - current_vocab_size)
    
    merge_loop_total = 0.0
    num_iters = 0
    for _ in range(max_merges):
        iter_start = time.perf_counter()
        pair_counts = collections.Counter()
        # For each token sequence (each word), count adjacent pairs.
        for seq, f in symbol_seq_freq.items():
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pair_counts[pair] += f
        if not pair_counts:
            break
        best_pair, best_pair_freq = min(pair_counts.items(), key=lambda x: (-x[1], x[0]))
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
        #print(f"Iteration {num_iters}: best pair {best_pair} (freq={best_pair_freq}) merged in {iter_time:.4f} seconds")
        if current_vocab_size >= vocab_size:
            break
    print(f"Total merge loop time for {num_iters} iterations: {merge_loop_total:.4f} seconds")
    
    # 5. Build final vocabulary.
    t_vocab_start = time.perf_counter()
    vocab = {}
    idx = 0
    # Add special tokens.
    for sp in special_tokens:
        vocab[idx] = sp.encode("utf-8")
        idx += 1
    # Base tokens: for each byte (0-255), use our byte encoder.
    for b in range(256):
        token_str = BYTE_ENCODER[b]
        vocab[idx] = token_str.encode("utf-8")
        idx += 1
    # Merged tokens: convert each merged token (a tuple of ints) to bytes.
    for merge in merges_list:
        # Each merge is a tuple of two tokens, where each token is a tuple of ints.
        merged_bytes = convert_token(merge[0]) + convert_token(merge[1])
        vocab[idx] = merged_bytes
        idx += 1
    t_vocab_end = time.perf_counter()
    print(f"Building final vocabulary took: {t_vocab_end - t_vocab_start:.4f} seconds")
    
    total_time = time.perf_counter() - t0
    print(f"Total training time: {total_time:.4f} seconds")
    
    # For testing, convert merges_list to a list of tuples of bytes.
    converted_merges = [(convert_token(a), convert_token(b)) for (a, b) in merges_list]
    
    return vocab, converted_merges