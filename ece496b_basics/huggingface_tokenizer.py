from tokenizers import ByteLevelBPETokenizer
from typing import List, Iterator, Iterable, Optional

class HuggingFaceTokenizer:
    def __init__(self, vocab_file: str, merges_file: str, special_tokens: Optional[List[str]] = None):
        # Initialize the HF byte-level BPE tokenizer from files.
        self.hf_tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        # Add any special tokens.
        if special_tokens:
            self.hf_tokenizer.add_special_tokens(special_tokens)
        # (Optional) You can adjust parameters if needed:
        # For GPT-2, the default settings of ByteLevelBPETokenizer are appropriate.

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs."""
        return self.hf_tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back into a string."""
        return self.hf_tokenizer.decode(ids)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings (such as lines from a file) into token IDs.
        This implementation simply iterates over the lines provided by the iterable.
        If your file is not newline-delimited, consider splitting into chunks yourself.
        """
        # If iterable is a file-like object (with a .read() method), iterate line-by-line.
        if hasattr(iterable, "read"):
            for line in iterable:
                for tid in self.encode(line):
                    yield tid
        else:
            for text in iterable:
                for tid in self.encode(text):
                    yield tid