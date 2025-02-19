import numpy as np
from huggingface_tokenizer import HuggingFaceTokenizer

# === CONFIGURATION: Update these paths as needed ===

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

# === Initialize the tokenizers ===
tinystories_tokenizer = HuggingFaceTokenizer(TINY_VOCAB, TINY_MERGES, special_tokens=SPECIAL_TOKENS)
openwebtext_tokenizer = HuggingFaceTokenizer(OWT_VOCAB, OWT_MERGES, special_tokens=SPECIAL_TOKENS)

# === Helper function to read a file and encode its contents ===
def encode_file(tokenizer, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    token_ids = tokenizer.encode(text)
    return token_ids

# === Encode each dataset ===
print("Encoding TinyStories training set...")
tiny_train_ids = encode_file(tinystories_tokenizer, TINY_TRAIN_PATH)
print("Encoding TinyStories development set...")
tiny_dev_ids = encode_file(tinystories_tokenizer, TINY_DEV_PATH)

print("Encoding OpenWebText training set...")
owt_train_ids = encode_file(openwebtext_tokenizer, OWT_TRAIN_PATH)
print("Encoding OpenWebText development set...")
owt_dev_ids = encode_file(openwebtext_tokenizer, OWT_DEV_PATH)

# === Convert to NumPy arrays with dtype uint16 ===
tiny_train_arr = np.array(tiny_train_ids, dtype=np.uint16)
tiny_dev_arr   = np.array(tiny_dev_ids, dtype=np.uint16)
owt_train_arr  = np.array(owt_train_ids, dtype=np.uint16)
owt_dev_arr    = np.array(owt_dev_ids, dtype=np.uint16)

# === Save the arrays to disk ===
np.save("/content/ECE491B-assignment1/serialized/tinystories_train.npy", tiny_train_arr)
np.save("/content/ECE491B-assignment1/serialized/tinystories_dev.npy", tiny_dev_arr)
np.save("/content/ECE491B-assignment1/serialized/openwebtext_train.npy", owt_train_arr)
np.save("/content/ECE491B-assignment1/serialized/openwebtext_dev.npy", owt_dev_arr)

print("Tokenization and serialization complete!")