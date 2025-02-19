import os
import random
from huggingface_tokenizer import HuggingFaceTokenizer

# --- CONFIGURATION: Update these paths as needed ---
# Paths to the vocabulary and merges files for each model.
TINYSTORIES_VOCAB = "/content/ECE491B-assignment1/bpe_tinystories/vocab.json"     # (10K vocabulary)
TINYSTORIES_MERGES = "/content/ECE491B-assignment1/bpe_tinystories/merges.txt"

OPENWEBTEXT_VOCAB = "/content/ECE491B-assignment1/owt/vocab.json"       # (32K vocabulary)
OPENWEBTEXT_MERGES = "/content/ECE491B-assignment1/owt/merges.txt"

# Directories containing document files for each corpus.
TINYSTORIES_DOCS_DIR = "/content/ECE491B-assignment1/data/TinyStoriesV2-GPT4-train.txt"          # should contain many text files
OPENWEBTEXT_DOCS_DIR = "/content/ECE491B-assignment1/data/owt_train.txt"            # should contain many text files

# Special tokens (if any)
SPECIAL_TOKENS = ["<|endoftext|>"]

# --- Initialize tokenizers ---
tinystories_tokenizer = HuggingFaceTokenizer(TINYSTORIES_VOCAB, TINYSTORIES_MERGES, special_tokens=SPECIAL_TOKENS)
openwebtext_tokenizer = HuggingFaceTokenizer(OPENWEBTEXT_VOCAB, OPENWEBTEXT_MERGES, special_tokens=SPECIAL_TOKENS)

# --- Utility function to compute compression ratio ---
def compute_compression_ratio(tokenizer, text):
    token_ids = tokenizer.encode(text)
    # Compression ratio: total input bytes (UTF-8) divided by number of tokens
    ratio = len(text.encode("utf-8")) / len(token_ids) if token_ids else 0
    return ratio, token_ids

# --- Sample 10 documents from each directory ---
def sample_documents(doc_dir, n=10):
    all_files = [f for f in os.listdir(doc_dir) if os.path.isfile(os.path.join(doc_dir, f))]
    # Ensure we have at least n files
    if len(all_files) < n:
        raise ValueError(f"Not enough files in {doc_dir} to sample {n} documents.")
    return random.sample(all_files, n)

tinystories_files = sample_documents(TINYSTORIES_DOCS_DIR, n=10)
openwebtext_files = sample_documents(OPENWEBTEXT_DOCS_DIR, n=10)

# --- Process TinyStories documents ---
tinystories_ratios = []
print("TinyStories Compression Ratios:")
for filename in tinystories_files:
    with open(os.path.join(TINYSTORIES_DOCS_DIR, filename), "r", encoding="utf-8") as f:
        text = f.read()
    ratio, _ = compute_compression_ratio(tinystories_tokenizer, text)
    tinystories_ratios.append(ratio)
    print(f"  {filename}: {ratio:.2f} bytes/token")
avg_tinystories_ratio = sum(tinystories_ratios) / len(tinystories_ratios)

# --- Process OpenWebText documents ---
openwebtext_ratios = []
print("\nOpenWebText Compression Ratios:")
for filename in openwebtext_files:
    with open(os.path.join(OPENWEBTEXT_DOCS_DIR, filename), "r", encoding="utf-8") as f:
        text = f.read()
    ratio, _ = compute_compression_ratio(openwebtext_tokenizer, text)
    openwebtext_ratios.append(ratio)
    print(f"  {filename}: {ratio:.2f} bytes/token")
avg_openwebtext_ratio = sum(openwebtext_ratios) / len(openwebtext_ratios)

# --- Summary ---
print("\nSummary:")
print(f"Average TinyStories compression ratio: {avg_tinystories_ratio:.2f} bytes/token")
print(f"Average OpenWebText compression ratio: {avg_openwebtext_ratio:.2f} bytes/token")