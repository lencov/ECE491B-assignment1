import os
import numpy as np
from huggingface_tokenizer import HuggingFaceTokenizer

# === CONFIGURATION: OpenWebText paths ===
OWT_VOCAB = "/content/ECE491B-assignment1/owt/vocab.json"
OWT_MERGES = "/content/ECE491B-assignment1/owt/merges.txt"
OWT_TRAIN_PATH = "/content/ECE491B-assignment1/data/owt_train.txt"

# Output prefix for the flushed parts.
OUT_PREFIX = "/content/ECE491B-assignment1/serialized/openwebtext_train"

# Special tokens (if any)
SPECIAL_TOKENS = ["<|endoftext|>"]

# Initialize the OpenWebText tokenizer.
openwebtext_tokenizer = HuggingFaceTokenizer(OWT_VOCAB, OWT_MERGES, special_tokens=SPECIAL_TOKENS)

def encode_and_flush_dataset(tokenizer, file_path, out_file_prefix, chunk_size=4096, flush_threshold=10_000_000):
    """
    Process a large file in chunks, tokenizing the text and flushing the token IDs
    to disk whenever the in-memory list reaches flush_threshold tokens.
    """
    token_ids = []
    leftover = ""
    total_bytes = 0
    part_number = 0

    os.makedirs(os.path.dirname(out_file_prefix), exist_ok=True)

    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk.encode("utf-8"))
            data = leftover + chunk

            # To avoid cutting a token in half, find the last whitespace.
            split_idx = data.rfind(" ")
            if split_idx == -1:
                # No whitespace found: process entire chunk.
                split_idx = len(data)
                leftover = ""
            else:
                leftover = data[split_idx:]
                data = data[:split_idx]
            
            token_ids.extend(tokenizer.encode(data))
            print(f"Processed {total_bytes} bytes, total tokens so far: {len(token_ids)}", flush=True)

            # Flush to disk if token_ids exceed the threshold.
            if len(token_ids) >= flush_threshold:
                part_filename = f"{out_file_prefix}_part{part_number}.npy"
                np.save(part_filename, np.array(token_ids, dtype=np.uint16))
                print(f"Flushed part {part_number} with {len(token_ids)} tokens to {part_filename}", flush=True)
                part_number += 1
                token_ids = []  # Clear the list to free memory

    # Process any remaining text.
    if leftover:
        token_ids.extend(tokenizer.encode(leftover))
    
    # Flush any remaining tokens.
    if token_ids:
        part_filename = f"{out_file_prefix}_part{part_number}.npy"
        np.save(part_filename, np.array(token_ids, dtype=np.uint16))
        print(f"Flushed final part {part_number} with {len(token_ids)} tokens to {part_filename}", flush=True)

    print("Finished encoding and flushing dataset.")

# Run the function on the OpenWebText training file.
encode_and_flush_dataset(openwebtext_tokenizer, OWT_TRAIN_PATH, OUT_PREFIX)