import regex as re
from typing import Dict, List, Tuple, Iterable, Iterator, Optional

class Tokenizer:
    # Default regex used for pre‐tokenization.
    DEFAULT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ):
        """
        Construct a tokenizer given a vocabulary, merges, and (optionally) special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        # Build reverse mapping: token bytes -> token ID.
        self.token_to_id = {token: tid for tid, token in self.vocab.items()}

        # Ensure all special tokens are in the vocabulary.
        for sp in self.special_tokens:
            sp_bytes = sp.encode('utf-8')
            if sp_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = sp_bytes
                self.token_to_id[sp_bytes] = new_id

        # Build a regex that first matches any special token (exactly) and then falls back to the default.
        if self.special_tokens:
            # Use non-capturing group so that re.findall returns the full match.
            special_tokens_pattern = '|'.join(map(re.escape, self.special_tokens))
            pattern = f"(?:{special_tokens_pattern})|{self.DEFAULT_PAT}"
        else:
            pattern = self.DEFAULT_PAT

        self.tokenizer_pattern = re.compile(pattern, re.UNICODE)

    def _encode_token(self, token: str) -> List[int]:
        """
        Convert a single pre-token (string) into one or more token IDs.
        """
        if token in self.special_tokens:
            return [self.token_to_id[token.encode('utf-8')]]
        else:
            pt_bytes = token.encode('utf-8')
            token_seq = [bytes([b]) for b in pt_bytes]
            for merge in self.merges:
                token_seq = self._apply_merge(token_seq, merge)
            return [self.token_to_id[t] for t in token_seq]

    def encode(self, text: str) -> List[int]:
        """
        Encode the entire input text (string) into a list of token IDs.
        """
        encoded_ids = []
        # Get all pre-tokens using our custom regex.
        pre_tokens = self.tokenizer_pattern.findall(text)
        for token in pre_tokens:
            encoded_ids.extend(self._encode_token(token))
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings into token IDs.
        When the iterable is a file-like object (i.e. has .read()),
        we read in fixed-size chunks and carefully handle tokens that
        might be split across chunk boundaries.
        """
        # If iterable is a file-like object, do chunked processing.
        if hasattr(iterable, "read"):
            buffer = ""
            chunk_size = 4096  # Adjust chunk size as needed.
            while True:
                chunk = iterable.read(chunk_size)
                if not chunk:
                    # End of file: process whatever is left.
                    for tid in self.encode(buffer):
                        yield tid
                    break
                buffer += chunk
                # Process the buffer into tokens.
                tokens, last_end = self._yield_tokens_from_buffer(buffer, final=False)
                for token in tokens:
                    for tid in self._encode_token(token):
                        yield tid
                # Leave the (possibly partial) last token in the buffer.
                buffer = buffer[last_end:]
        else:
            # For other iterables assume each string is small enough.
            for text in iterable:
                for tid in self.encode(text):
                    yield tid

    def _yield_tokens_from_buffer(self, buffer: str, final: bool) -> Tuple[List[str], int]:
        """
        Tokenize the current buffer using our regex. If not in final mode,
        and if the last match ends exactly at the end of the buffer, assume
        it might be incomplete and leave it for the next chunk.
        Returns a tuple (list_of_tokens, last_processed_index).
        """
        tokens = []
        last_end = 0
        for m in self.tokenizer_pattern.finditer(buffer):
            # If we're not final and this match goes to the very end,
            # then it might be incomplete—stop here.
            if not final and m.end() == len(buffer):
                break
            tokens.append(m.group(0))
            last_end = m.end()
        return tokens, last_end

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a string.
        """
        byte_seq = b"".join(self.vocab[tid] for tid in ids if tid in self.vocab)
        return byte_seq.decode('utf-8', errors='replace')

    def _apply_merge(self, token_seq: List[bytes], merge: Tuple[bytes, bytes]) -> List[bytes]:
        """
        Apply a single BPE merge to a token sequence.
        """
        new_seq = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == merge:
                new_seq.append(token_seq[i] + token_seq[i+1])
                i += 2  # Skip the next token as it was merged.
            else:
                new_seq.append(token_seq[i])
                i += 1
        return new_seq