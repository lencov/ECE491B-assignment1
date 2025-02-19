import os
import numpy as np
from huggingface_tokenizer import HuggingFaceTokenizer

# === CONFIGURATION: Correct paths ===
# TinyStories paths (10K vocabulary)
TINY_VOCAB = "/content/ECE491B-assignment1/bpe_tinystories/vocab.json"
TINY_MERGES = "/content/ECE491B-assignment1/bpe_tinystories/merges.txt"
TINY_TRAIN_PATH = "/content/ECE491B-assignment1/data/TinyStoriesV2-GPT4-train.txt"
TINY_DEV_PATH   = "/content/ECE491B-assignment1/data/TinyStoriesV2-GPT4-valid.txt"

# OpenWebText paths (32K vocabulary)
OWT_VOCAB = "/content/ECE491B-assignment1/owt/vocab.json"
OWT_MERGES = "/content/ECE491B-assignment1/owt/merges.txt"
OWT_TRAIN_PATH = "/content/ECE491B-assignment1/data/owt_train.txt"
OWT_DEV_PATH   = "/content/ECE491B-assignment1/data/owt_valid.txt"

# Special tokens (if any)
SPECIAL_TOKENS = ["<|endoftext|>"]

# === Initialize Tokenizers ===
tinystories_tokenizer = HuggingFaceTokenizer(TINY_VOCAB, TINY_MERGES, special_tokens=SPECIAL_TOKENS)
openwebtext_tokenizer = HuggingFaceTokenizer(OWT_VOCAB, OWT_MERGES, special_tokens=SPECIAL_TOKENS)

# === Helper: Process file in chunks with progress logging ===
def encode_file_in_chunks(tokenizer, file_path, chunk_size=4096):
    """
    Reads the file in chunks, appending any leftover text to avoid cutting tokens,
    and logs progress in terms of bytes processed.
    """
    token_ids = []
    leftover = ""
    total_bytes = 0

    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk.encode("utf-8"))
            data = leftover + chunk

            # To avoid splitting tokens, try to find the last whitespace.
            split_idx = data.rfind(" ")
            if split_idx == -1:
                # If no whitespace found, process the entire chunk.
                split_idx = len(data)
                leftover = ""
            else:
                leftover = data[split_idx:]
                data = data[:split_idx]
            
            token_ids.extend(tokenizer.encode(data))
            print(f"Processed {total_bytes} bytes...", flush=True)

    # Process any remaining text.
    if leftover:
        token_ids.extend(tokenizer.encode(leftover))
    return token_ids

def encode_dataset(tokenizer, file_path, dataset_name):
    print(f"Encoding {dataset_name} from {file_path} ...")
    ids = encode_file_in_chunks(tokenizer, file_path)
    print(f"Finished encoding {dataset_name}. Total tokens: {len(ids)}")
    return ids

# === Encode each dataset ===
tiny_train_ids = encode_dataset(tinystories_tokenizer, TINY_TRAIN_PATH, "TinyStories Train")
tiny_dev_ids   = encode_dataset(tinystories_tokenizer, TINY_DEV_PATH, "TinyStories Dev")
owt_train_ids  = encode_dataset(openwebtext_tokenizer, OWT_TRAIN_PATH, "OpenWebText Train")
owt_dev_ids    = encode_dataset(openwebtext_tokenizer, OWT_DEV_PATH, "OpenWebText Dev")

# === Convert to NumPy arrays of dtype uint16 ===
tiny_train_arr = np.array(tiny_train_ids, dtype=np.uint16)
tiny_dev_arr   = np.array(tiny_dev_ids, dtype=np.uint16)
owt_train_arr  = np.array(owt_train_ids, dtype=np.uint16)
owt_dev_arr    = np.array(owt_dev_ids, dtype=np.uint16)

# === Save the arrays ===
os.makedirs("/content/ECE491B-assignment1/serialized", exist_ok=True)
np.save("/content/ECE491B-assignment1/serialized/tinystories_train.npy", tiny_train_arr)
np.save("/content/ECE491B-assignment1/serialized/tinystories_dev.npy", tiny_dev_arr)
np.save("/content/ECE491B-assignment1/serialized/openwebtext_train.npy", owt_train_arr)
np.save("/content/ECE491B-assignment1/serialized/openwebtext_dev.npy", owt_dev_arr)

print("Tokenization and serialization complete!")