import time
import psutil
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

# Function to measure memory usage
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)  # Convert to GB

# Define input file and output paths
input_path = "/content/ECE491B-assignment1/data/owt_train.txt"  # Update this to the actual dataset path
vocab_size = 32000
special_tokens = ["<|endoftext|>"]

# Track time and memory
start_time = time.time()
start_mem = get_memory_usage()

# Initialize the tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer
print("Training BPE tokenizer with Hugging Face tokenizers...")
tokenizer.train(files=[input_path], vocab_size=vocab_size, special_tokens=special_tokens)
print("BPE tokenizer training complete.")

# Measure end time and memory
end_time = time.time()
end_mem = get_memory_usage()

# Save the tokenizer to disk
output_dir = Path("owt")
output_dir.mkdir(exist_ok=True)
tokenizer.save_model(str(output_dir))

# Find the longest token in the vocabulary
vocab = tokenizer.get_vocab()
longest_token = max(vocab.keys(), key=len)

# Print results
print(f"Training Time: {end_time - start_time:.2f} seconds")
print(f"Memory Usage: {end_mem - start_mem:.2f} GB")
print(f"Longest Token: {longest_token} ({len(longest_token)} bytes)")