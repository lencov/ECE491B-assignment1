import regex as re
import collections
from pathlib import Path
import time
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# --- Byte encoder: Map byte values (0-255) to Unicode characters.
def bytes_to_unicode():
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

BYTE_ENCODER = bytes_to_unicode()

def convert_token(token_tuple, byte_encoder=BYTE_ENCODER):
    try:
        s = "".join(byte_encoder[b] for b in token_tuple)
    except TypeError:
        print(f"Warning: Expected token_tuple to be iterable, got: {token_tuple}", file=sys.stderr)
        s = ""
    return s.encode("utf-8")

# --- Token streaming with batch overlap
def findall_with_timeout(pattern, chunk, timeout=5):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pattern.findall, chunk)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return None

def stream_tokens(text, pattern, batch_size=100000, overlap=50, timeout=5):
    """
    Generator that yields tokens from text in batches with overlap.
    """
    text_len = len(text)
    for start in range(0, text_len, batch_size - overlap):
        end = min(start + batch_size, text_len)
        chunk = text[start:end]
        chunk_tokens = findall_with_timeout(pattern, chunk, timeout=timeout)
        if chunk_tokens is None:
            print(f"⚠️ Timeout processing chunk starting at {start} (batch size {batch_size}).", file=sys.stderr)
            continue
        for token in chunk_tokens:
            token_bytes = token.encode("utf-8", errors="strict")
            yield tuple((b,) for b in token_bytes)  # Ensure each byte is wrapped in a tuple

# --- BPE Training
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    1) Read text from file.
    2) Pre-tokenize the text in batches (streaming tokens with overlap).
    3) Build a frequency counter directly from the token generator.
    4) Iteratively merge tokens via BPE until the vocabulary size is reached.
    5) Build the final vocabulary.
    """
    t0 = time.perf_counter()

    if special_tokens is None:
        special_tokens = []

    # 1. Read text
    path = Path(input_path)
    text = path.read_text(encoding="utf-8")
    
    # 2. Pre-tokenization using GPT-2 regex (compiled once)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)
    token_generator = stream_tokens(text, pattern, batch_size=100000, overlap=50, timeout=5)

    # 3. Build frequency counter from the token generator.
    freq_dict = collections.Counter(token for token in token_generator)

    # 4. Symbol sequence frequency mapping (start with base frequency counts)
    symbol_seq_freq = dict(freq_dict)

    merges_list = []
    current_vocab_size = 256 + len(special_tokens)
    max_merges = max(0, vocab_size - current_vocab_size)

    for _ in range(max_merges):
        pair_counts = collections.Counter()
        for seq, f in symbol_seq_freq.items():
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pair_counts[pair] += f

        if not pair_counts:
            break

        # 5. Correct tie-breaking: Use the lexicographically greater pair
        best_pair, best_pair_freq = max(pair_counts.items(), key=lambda x: (x[1], x[0]))

        if best_pair_freq == 0:
            break

        # Store merges in lexicographical order
        merges_list.append(tuple(sorted(best_pair, reverse=True)))

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

        if current_vocab_size >= vocab_size:
            break

    # 6. Build final vocabulary
    vocab = {}
    idx = 0

    # Add special tokens
    for sp in special_tokens:
        vocab[idx] = sp.encode("utf-8")
        idx += 1

    # Base tokens (bytes 0-255)
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1

    # Add merged tokens
    for merge in merges_list:
        merged_bytes = convert_token(merge[0]) + convert_token(merge[1])
        vocab[idx] = merged_bytes
        idx += 1

    total_time = time.perf_counter() - t0
    print(f"Total training time: {total_time:.4f} seconds")

    return vocab, merges_list