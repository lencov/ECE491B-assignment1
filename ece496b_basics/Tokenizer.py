import regex as re
from typing import Dict, List, Tuple, Iterable, Iterator, Optional

class Tokenizer:
    # Base pattern from GPT‑2, which (by default) may add an optional leading space.
    BASE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ):
        """
        vocab: mapping from token ID (int) to token bytes.
        merges: list of BPE merges (each a tuple of two bytes objects).
        special_tokens: list of special tokens as strings.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        # Build a reverse mapping from token bytes to token ID.
        self.token_to_id = {token: tid for tid, token in self.vocab.items()}

        # Ensure that every special token is present in the vocabulary.
        for sp in self.special_tokens:
            sp_bytes = sp.encode("utf-8")
            if sp_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = sp_bytes
                self.token_to_id[sp_bytes] = new_id

        # If we have special tokens, build a pattern that captures them.
        if self.special_tokens:
            # Sort special tokens by length (longest first) so that overlapping tokens work correctly.
            special_pattern = "|".join(
                map(re.escape, sorted(self.special_tokens, key=len, reverse=True))
            )
            # Use a capturing group for special tokens; then try the base pattern.
            pattern = f"({special_pattern})|" + self.BASE_PATTERN
        else:
            pattern = self.BASE_PATTERN

        self.pattern = re.compile(pattern)

    def encode(self, text: str) -> List[int]:
        """
        Encode an input string into a list of token IDs.
        Special tokens that appear exactly in the text will be used as a whole.
        Otherwise the text is encoded as UTF-8 bytes and merged via BPE.
        """
        encoded_ids = []
        for m in self.pattern.finditer(text):
            # If group(1) is not None, then a special token was matched.
            token = m.group(1) if m.group(1) is not None else m.group(0)
            if token in self.special_tokens:
                # Special token: convert to bytes and look up its ID.
                sp_bytes = token.encode("utf-8")
                encoded_ids.append(self.token_to_id[sp_bytes])
            else:
                # For a regular token, convert to bytes.
                pt_bytes = token.encode("utf-8")
                # Start with single-byte tokens.
                token_seq = [bytes([b]) for b in pt_bytes]
                # Apply each BPE merge in order.
                for merge in self.merges:
                    token_seq = self._apply_merge(token_seq, merge)
                # Look up each resulting token in the vocabulary.
                for t in token_seq:
                    if t in self.token_to_id:
                        encoded_ids.append(self.token_to_id[t])
                    else:
                        raise ValueError(f"Token {t} not found in vocabulary.")
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings (e.g. lines from a file) into token IDs.
        """
        for text in iterable:
            for tid in self.encode(text):
                yield tid

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a string.
        Concatenates the token bytes and decodes as UTF-8.
        """
        byte_seq = b"".join(self.vocab[tid] for tid in ids if tid in self.vocab)
        return byte_seq.decode("utf-8", errors="replace")

    def _apply_merge(self, token_seq: List[bytes], merge: Tuple[bytes, bytes]) -> List[bytes]:
        """
        Apply a single BPE merge to a token sequence.
        Scans for an adjacent pair that exactly matches the merge and replaces them
        with their concatenation.
        """
        new_seq = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == merge:
                new_seq.append(token_seq[i] + token_seq[i+1])
                i += 2  # Skip the next token since it was merged.
            else:
                new_seq.append(token_seq[i])
                i += 1
        return new_seq