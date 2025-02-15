import regex as re
import ast
from typing import List, Iterator, Iterable, Tuple, Dict

class Tokenizer:
    # GPT-2 style regex for pre-tokenization.
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        vocab: mapping from token ID (int) to token bytes.
        merges: list of BPE merges (each a tuple of two bytes objects).
        special_tokens: list of special tokens as strings.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        # Build a reverse mapping from token bytes to token ID.
        self.token_to_id = {token: tid for tid, token in self.vocab.items()}

        # Ensure that all special tokens are in the vocabulary.
        for sp in self.special_tokens:
            sp_bytes = sp.encode('utf-8')
            if sp_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = sp_bytes
                self.token_to_id[sp_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """
        Construct a Tokenizer from a serialized vocabulary and merges file.
        Assumes the files contain Python literal representations of the data.
        """
        # Load vocabulary.
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            vocab_str = vf.read()
            vocab = ast.literal_eval(vocab_str)
            # Ensure that all values are bytes.
            for k, v in vocab.items():
                if isinstance(v, str):
                    vocab[k] = v.encode('utf-8')
        # Load merges.
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            merges_str = mf.read()
            merges = ast.literal_eval(merges_str)
            # Ensure each merge is a tuple of bytes.
            merges = [
                (
                    a if isinstance(a, bytes) else a.encode('utf-8'),
                    b if isinstance(b, bytes) else b.encode('utf-8')
                )
                for a, b in merges
            ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encode an input string into a sequence of token IDs.
        Steps:
          1. Pre-tokenize using the regex pattern.
          2. For each pre-token:
             a. If it exactly matches a special token, use that token ID.
             b. Otherwise, convert the string to UTF-8 bytes,
                break it into a list of single-byte tokens,
                then apply the learned BPE merges in order.
          3. Map each final token (bytes) to its vocabulary ID.
        """
        encoded_ids = []
        # Pre-tokenize the input text.
        pre_tokens = re.findall(self.PAT, text)
        for pt in pre_tokens:
            # Check for an exact match to a special token.
            if pt in self.special_tokens:
                sp_bytes = pt.encode('utf-8')
                encoded_ids.append(self.token_to_id[sp_bytes])
            else:
                # Convert pre-token to its byte representation.
                pt_bytes = pt.encode('utf-8')
                # Initialize the token sequence as a list of single-byte tokens.
                token_seq = [bytes([b]) for b in pt_bytes]
                # Apply BPE merges in the learned order.
                for merge in self.merges:
                    token_seq = self._apply_merge(token_seq, merge)
                # Convert each merged token into its corresponding ID.
                for token in token_seq:
                    if token in self.token_to_id:
                        encoded_ids.append(self.token_to_id[token])
                    else:
                        # This should not happen if training and encoding use the same vocab.
                        raise ValueError(f"Token {token} not found in vocabulary.")
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings (for example, lines from a file) into token IDs.
        This is useful for processing large texts without loading the entire content into memory.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back into a string.
        Concatenate the vocabulary byte strings corresponding to the token IDs
        and decode using UTF-8. If decoding fails, replace malformed data with U+FFFD.
        """
        # Concatenate the bytes for all token IDs.
        byte_seq = b"".join(self.vocab[tid] for tid in ids if tid in self.vocab)
        return byte_seq.decode('utf-8', errors='replace')

    def _apply_merge(self, token_seq: List[bytes], merge: Tuple[bytes, bytes]) -> List[bytes]:
        """
        Apply a single merge to a sequence of tokens.
        Scans the token sequence for any adjacent pair that exactly matches `merge`
        and replaces them with their concatenation.
        """
        new_seq = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == merge:
                # Merge the two tokens.
                new_seq.append(token_seq[i] + token_seq[i+1])
                i += 2  # Skip the next token since it was merged.
            else:
                new_seq.append(token_seq[i])
                i += 1
        return new_seq