import time
import psutil
import json
from pathlib import Path
from ece496b_basics.bpe_trainer import train_bpe

# Function to measure memory usage
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)  # Convert to GB

# Define input file and output paths
input_path = "/content/ECE491B-assignment1/data/TinyStoriesV2-GPT4-train.txt"  # Update this to the actual dataset path
vocab_size = 10000
special_tokens = ["<|endoftext|>"]

# Track time and memory
start_time = time.time()
start_mem = get_memory_usage()

# Train BPE tokenizer
print("Training BPE tokenizer...")
vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
print("BPE tokenizer training complete.")
# Measure end time and memory
end_time = time.time()
end_mem = get_memory_usage()

# Save vocab and merges to disk
output_dir = Path("bpe_tinystories")
output_dir.mkdir(exist_ok=True)
with open(output_dir / "vocab.json", "w") as f:
    json.dump({k: v.decode("utf-8", errors="ignore") for k, v in vocab.items()}, f, indent=2)

with open(output_dir / "merges.txt", "w") as f:
    for merge in merges:
        f.write(f"{bytes(merge[0]).decode('utf-8', errors='ignore')} {bytes(merge[1]).decode('utf-8', errors='ignore')}\n")

# Find the longest token in the vocabulary
longest_token = max(vocab.values(), key=len)

# Print results
print(f"Training Time: {end_time - start_time:.2f} seconds")
print(f"Memory Usage: {end_mem - start_mem:.2f} GB")
print(f"Longest Token: {longest_token.decode('utf-8', errors='ignore')} ({len(longest_token)} bytes)")