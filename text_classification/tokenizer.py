from collections import Counter, defaultdict
from pathlib import Path
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import spacy
from spacy.symbols import ORTH

from rules import default_rules


BOS, FLD, UNK, PAD = SPECIAL_TOKENS = 'xxbox', 'xxfld', 'xxunk', 'xxpad'


class SpacyTokenizer:
    """A thin wrapper on top of Spacy tokenization tools."""

    def __init__(self, lang='en', rules=default_rules, special_tokens=SPECIAL_TOKENS):
        tokenizer = spacy.load(lang).tokenizer
        if special_tokens:
            for token in special_tokens:
                tokenizer.add_special_case(token, [{ORTH: token}])

        self.rules = rules or []
        self.tokenizer = tokenizer

    def tokenize(self, text: str):
        """Converts a single string into list of tokens."""

        for rule in self.rules:
            text = rule(text)
        return [t.text for t in self.tokenizer(text)]


class Vocab:

    def __init__(self, itos):
        self.itos = itos
        self.stoi = defaultdict(int, {v: k for k, v in enumerate(itos)})
        self.size = len(itos)

    def __eq__(self, other):
        if not isinstance(other, Vocab):
            raise TypeError(
                'can only compare with another Vocab instance, '
                'got %s' % type(other))
        return self.itos == other.itos

    def save(self, path: Path):
        with path.open('wb') as file:
            pickle.dump(self.itos, file)

    @staticmethod
    def load(path: Path) -> 'Vocab':
        with path.open('rb') as file:
            itos = pickle.load(file)
        return Vocab(itos)

    @staticmethod
    def make_vocab(tokens, min_freq: int=3, max_vocab: int=60000, pad=PAD, unknown=UNK) -> 'Vocab':
        freq = Counter(token for sentence in tokens for token in sentence)
        most_common = freq.most_common(max_vocab)
        itos = [token for token, count in most_common if count > min_freq]
        itos.insert(0, pad)
        if unknown in itos:
            itos.remove(unknown)
        itos.insert(0, unknown)
        return Vocab(itos)

    def numericalize(self, texts):
        return [
            np.array([self.stoi[token] for token in text], dtype=np.int)
            for text in texts]

    def textify_all(self, samples):
        return [self.textify(sample) for sample in samples]

    def textify(self, tokens):
        return ' '.join([self.itos[number] for number in tokens])


def tokenize_in_parallel(texts):
    n_workers = cpu_count()
    parts = split_into(texts, len(texts)//n_workers + 1)
    with Pool(n_workers) as pool:
        results = pool.map(tokenize, parts)
    return sum(results, [])


def tokenize(texts):
    tokenizer = SpacyTokenizer()
    return [tokenizer.tokenize(text) for text in texts]


def split_into(arr, n):
    return [arr[i:i + n] for i in range(0, len(arr), n)]
