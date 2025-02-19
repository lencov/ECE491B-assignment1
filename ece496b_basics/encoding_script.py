import os
import random
from huggingface_tokenizer import HuggingFaceTokenizer

# --- CONFIGURATION: Update these paths as needed ---
# Paths to the vocabulary and merges files for each model.
TINYSTORIES_VOCAB = "/content/ECE491B-assignment1/bpe_tinystories/vocab.json"     # (10K vocabulary)
TINYSTORIES_MERGES = "/content/ECE491B-assignment1/bpe_tinystories/merges.txt"

OPENWEBTEXT_VOCAB = "/content/ECE491B-assignment1/owt/vocab.json"       # (32K vocabulary)
OPENWEBTEXT_MERGES = "/content/ECE491B-assignment1/owt/merges.txt"

# Paths to document collections.
# If the path is a file, the script will attempt to split it into documents using double newlines.
TINYSTORIES_DOCS_PATH = "/content/ECE491B-assignment1/data/TinyStoriesV2-GPT4-train.txt"
OPENWEBTEXT_DOCS_PATH = "/content/ECE491B-assignment1/data/owt_train.txt"

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

# --- Modified sampling function ---
def sample_documents(doc_path, n=10):
    """
    If doc_path is a directory, sample n files.
    If it's a file, split its content into documents using double newlines as delimiter and sample n documents.
    """
    if os.path.isdir(doc_path):
        all_files = [f for f in os.listdir(doc_path) if os.path.isfile(os.path.join(doc_path, f))]
        if len(all_files) < n:
            raise ValueError(f"Not enough files in {doc_path} to sample {n} documents.")
        selected = random.sample(all_files, n)
        docs = []
        for filename in selected:
            with open(os.path.join(doc_path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
        return docs
    elif os.path.isfile(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Split on double newline; adjust delimiter if needed.
        docs = [doc.strip() for doc in text.split("\n\n") if doc.strip()]
        if len(docs) < n:
            raise ValueError(f"Not enough documents in file {doc_path} to sample {n} documents.")
        return random.sample(docs, n)
    else:
        raise ValueError(f"{doc_path} is not a valid file or directory.")

# --- Sample 10 documents from each source ---
tinystories_docs = sample_documents(TINYSTORIES_DOCS_PATH, n=10)
openwebtext_docs = sample_documents(OPENWEBTEXT_DOCS_PATH, n=10)

# --- Process TinyStories documents ---
tinystories_ratios = []
print("TinyStories Compression Ratios:")
for i, doc in enumerate(tinystories_docs):
    ratio, _ = compute_compression_ratio(tinystories_tokenizer, doc)
    tinystories_ratios.append(ratio)
    print(f"  Document {i+1}: {ratio:.2f} bytes/token")
avg_tinystories_ratio = sum(tinystories_ratios) / len(tinystories_ratios)

# --- Process OpenWebText documents ---
openwebtext_ratios = []
print("\nOpenWebText Compression Ratios:")
for i, doc in enumerate(openwebtext_docs):
    ratio, _ = compute_compression_ratio(openwebtext_tokenizer, doc)
    openwebtext_ratios.append(ratio)
    print(f"  Document {i+1}: {ratio:.2f} bytes/token")
avg_openwebtext_ratio = sum(openwebtext_ratios) / len(openwebtext_ratios)

# --- Summary ---
print("\nSummary:")
print(f"Average TinyStories compression ratio: {avg_tinystories_ratio:.2f} bytes/token")
print(f"Average OpenWebText compression ratio: {avg_openwebtext_ratio:.2f} bytes/token")