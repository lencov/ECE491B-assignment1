import os
import random
from huggingface_tokenizer import HuggingFaceTokenizer

# --- CONFIGURATION ---
TINYSTORIES_VOCAB = "/content/ECE491B-assignment1/bpe_tinystories/vocab.json"     # (10K vocabulary)
TINYSTORIES_MERGES = "/content/ECE491B-assignment1/bpe_tinystories/merges.txt"

OPENWEBTEXT_VOCAB = "/content/ECE491B-assignment1/owt/vocab.json"       # (32K vocabulary)
OPENWEBTEXT_MERGES = "/content/ECE491B-assignment1/owt/merges.txt"

# Use file paths for document sources.
TINYSTORIES_DOCS_PATH = "/content/ECE491B-assignment1/data/TinyStoriesV2-GPT4-train.txt"
OPENWEBTEXT_DOCS_PATH = "/content/ECE491B-assignment1/data/owt_train.txt"

SPECIAL_TOKENS = ["<|endoftext|>"]

# --- Initialize tokenizers ---
tinystories_tokenizer = HuggingFaceTokenizer(TINYSTORIES_VOCAB, TINYSTORIES_MERGES, special_tokens=SPECIAL_TOKENS)
openwebtext_tokenizer = HuggingFaceTokenizer(OPENWEBTEXT_VOCAB, OPENWEBTEXT_MERGES, special_tokens=SPECIAL_TOKENS)

# --- Utility function to compute compression ratio ---
def compute_compression_ratio(tokenizer, text):
    token_ids = tokenizer.encode(text)
    # Compression ratio: input bytes (UTF-8) divided by number of tokens.
    ratio = len(text.encode("utf-8")) / len(token_ids) if token_ids else 0
    return ratio, token_ids

# --- Sampling Functions ---
def sample_documents(doc_path, n=10):
    """
    If doc_path is a directory, sample n files.
    If it's a file, use reservoir sampling to pick n documents.
    Documents are assumed to be separated by a blank line.
    """
    if os.path.isdir(doc_path):
        all_files = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if os.path.isfile(os.path.join(doc_path, f))]
        if len(all_files) < n:
            raise ValueError(f"Not enough files in {doc_path} to sample {n} documents.")
        docs = []
        selected = random.sample(all_files, n)
        for filename in selected:
            with open(filename, "r", encoding="utf-8") as f:
                docs.append(f.read())
        return docs
    elif os.path.isfile(doc_path):
        return sample_documents_from_file(doc_path, n)
    else:
        raise ValueError(f"{doc_path} is not a valid file or directory.")

def sample_documents_from_file(doc_path, n=10):
    """
    Reads the file line by line, splitting documents on blank lines,
    and uses reservoir sampling to select n documents.
    """
    reservoir = []
    count = 0
    current_doc_lines = []
    
    with open(doc_path, "r", encoding="utf-8") as f:
        for line in f:
            # If the line is blank, consider it a document separator.
            if line.strip() == "":
                if current_doc_lines:
                    doc = "\n".join(current_doc_lines).strip()
                    count += 1
                    if len(reservoir) < n:
                        reservoir.append(doc)
                    else:
                        r = random.randint(0, count - 1)
                        if r < n:
                            reservoir[r] = doc
                    current_doc_lines = []
            else:
                current_doc_lines.append(line.rstrip("\n"))
        # Process any remaining lines as the final document.
        if current_doc_lines:
            doc = "\n".join(current_doc_lines).strip()
            count += 1
            if len(reservoir) < n:
                reservoir.append(doc)
            else:
                r = random.randint(0, count - 1)
                if r < n:
                    reservoir[r] = doc
                    
    if len(reservoir) < n:
        raise ValueError(f"Not enough documents in file {doc_path} to sample {n} documents.")
    return reservoir

# --- Sample Documents ---
tinystories_docs = sample_documents(TINYSTORIES_DOCS_PATH, n=10)
openwebtext_docs = sample_documents(OPENWEBTEXT_DOCS_PATH, n=10)

# --- Process TinyStories Documents ---
tinystories_ratios = []
print("TinyStories Compression Ratios:")
for i, doc in enumerate(tinystories_docs):
    ratio, _ = compute_compression_ratio(tinystories_tokenizer, doc)
    tinystories_ratios.append(ratio)
    print(f"  Document {i+1}: {ratio:.2f} bytes/token")
avg_tinystories_ratio = sum(tinystories_ratios) / len(tinystories_ratios)

# --- Process OpenWebText Documents ---
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