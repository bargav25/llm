import regex as re
import pickle
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Iterable, Iterator
from pathlib import Path


def merge(token: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """
    Merge occurrences of 'pair' in 'token' list with 'new_id'.
    """
    res = []
    i = 0
    while i < len(token):
        if i + 1 < len(token) and (token[i], token[i+1]) == pair:
            res.append(new_id)
            i += 2
        else:
            res.append(token[i])
            i += 1
    return res

class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer class with parallel file encoding.
    """

    def __init__(
        self,
        merges: Dict[Tuple[int, int], int],
        vocab: Dict[int, bytes],
        special_token_map: Dict[str, int]
    ):
        self.merges = merges
        self.vocab = vocab
        self.special_token_map = special_token_map
        # Pre-compile token regex using `regex` for \p unicode categories
        self.PAT = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        if special_token_map:
            pattern = '(' + '|'.join(map(re.escape, special_token_map.keys())) + ')'
            self.SPECIAL_PAT = re.compile(pattern)
        else:
            self.SPECIAL_PAT = None

    def encode(self, text: str) -> List[int]:
        """
        Encode a single string to a list of token IDs.
        """
        parts = [text]
        if self.SPECIAL_PAT:
            parts = self.SPECIAL_PAT.split(text)
        token_ids: List[int] = []

        # Initial tokenization into raw byte sequences
        tokens: List[List[int]] = []
        for part in parts:
            if self.SPECIAL_PAT and part in self.special_token_map:
                tokens.append([self.special_token_map[part]])
            else:
                for sub in self.PAT.findall(part):
                    tokens.append(list(sub.encode('utf-8')))

        # Apply BPE merges
        for token in tokens:
            for pair, new_id in self.merges.items():
                token = merge(token, pair, new_id)
            token_ids.extend(token)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode lines from an iterable, yielding token IDs one by one.
        """
        for line in iterable:
            line = line.strip()
            if not line:
                continue
            token_ids = self.encode(line)
            yield from token_ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back to a string.
        """
        byte_seq = b''.join(self.vocab[i] for i in ids)
        return byte_seq.decode('utf-8', errors='replace')

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    @classmethod
    def from_file(cls, filepath: str) -> 'BPETokenizer':
        """
        Load tokenizer data from a pickle file and return an instance.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        vocab = data['vocab']
        merges = data['merges']
        special = data.get('special_token_map', {})
        inst = cls(merges=merges, vocab=vocab, special_token_map=special)
        inst.tokenizer_file = filepath
        return inst

 

if __name__ == "__main__":

    tokenizer = BPETokenizer.from_file("output/bpe_tokenizer.pkl")

    ids = tokenizer("Hello, how you are doing ? <|endoftext|> I am sardar gabbarsingh haha!!, <|endoftext|> naam tho suna hoga ")

    print(tokenizer.decode(ids))



    