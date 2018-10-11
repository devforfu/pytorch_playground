from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import spacy
from spacy.symbols import ORTH

from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from rules import default_rules


IMDB = Path.home() / 'data' / 'aclImdb'
TRAIN_PATH = IMDB / 'train'
TEST_PATH = IMDB / 'test'
CLASSES = ['neg', 'pos', 'unsup']

BOS, FLD, UNK, PAD = SPECIAL_TOKENS = 'xxbox', 'xxfld', 'xxunk', 'xxpad'


def main():
    dataset = ImdbDataset(IMDB, tokenizer=tokenize_in_parallel)


class ImdbDataset:
    """Represents the IMDB movie reviews dataset.

    The dataset contains 50000 supervised, and 50000 unsupervised movie reviews
    with positive and negative sentiment ratings. The supervised subset of data
    is separated into two equally sized sets, with 12500 instances per class.

    The two flags, `supervised` and `train` define which subset of the data
    we're going to load. There are four possible cases:

    +-------+------------+--------+-------+---------+
    | Train | Supervised | Folder | Size  | Labels? |
    +-------+------------+--------+-------+---------+
    | True  | True       | train  | 25000 | Yes     |
    | False | True       | test   | 25000 | Yes     |
    | True  | False      | train  | 75000 | No      |
    | False | False      | test   | 25000 | No      |
    +-------+------------+--------+-------+---------+
    """
    def __init__(self, root: Path, train=True, supervised=False, tokenizer=None,
                 make_vocab=None):
        """
        Args:
             root: Path to the folder with train and tests subfolders.
             supervised: If True, then the data from supervised subset is loaded.
             train: If True, then the data from training subset is loaded.

        """
        self.root = root
        self.train = train
        self.supervised = supervised

        subfolder = root / ('train' if train else 'test')
        if tokenizer is None:
            tokenizer = lambda x: x

        if supervised:
            texts, labels = [], []
            for index, label in enumerate(CLASSES):
                for filename in (subfolder/label).glob('*.txt'):
                    texts.append(filename.open('r').read())
                    labels.append(index)
            if train:
                self.train_labels = labels
            else:
                self.test_labels = labels

        else:
            texts = []
            for label in CLASSES:
                files_folder = subfolder / label
                if files_folder.exists():
                    for filename in files_folder.glob('*.txt'):
                        texts.append(filename.open('r').read())

        tokens = tokenizer(texts)
        if make_vocab is not None:
            self.vocab = make_vocab(tokens)

        if train:
            self.train_data = tokens
        else:
            self.test_data = tokens

    def __getitem__(self, index):
        if self.train and self.supervised:
            return self.train_data[index], self.train_labels[index]
        elif self.train and not self.supervised:
            return self.train_data[index], None
        elif not self.train and self.supervised:
            return self.test_data[index], self.test_labels[index]
        else:
            return self.test_data[index], None

    def __len__(self):
        return len(self.train_data if self.train else self.test_data)


class SpacyTokenizer:
    """A thin wrapper on top of Spacy tokenization tools."""

    def __init__(self, lang='en', rules=default_rules, special_tokens=SPECIAL_TOKENS):
        tokenizer = spacy.load(lang).tokenizer
        if special_tokens:
            for token in special_tokens:
                tokenizer.add_special_tokens(token, [{ORTH: token}])

        self.rules = rules or []
        self.tokenizer = tokenizer

    def tokenize(self, text: str):
        """Converts a single string into list of tokens."""

        for rule in self.rules:
            text = rule(text)
        return [t.text for t in self.tokenizer(text)]


def tokenize_in_parallel(texts):

    def tokenize(subset):
        tokenizer = SpacyTokenizer()
        return [tokenizer.tokenize(text) for text in subset]

    n_workers = 1  # cpu_count()
    parts = split_into(texts, len(texts)//n_workers + 1)
    with Pool(n_workers) as pool:
        results = pool.map(tokenize, parts)
    return sum(results, [])


def split_into(arr, n):
    return [arr[i:i + n] for i in range(0, len(arr), n)]


class Vocab:

    def __init__(self, itos):
        self.itos = itos
        self.stoi = defaultdict(int, {v: k for k, v in enumerate(itos)})
        self.size = len(itos)

    def save(self, path: Path):
        with path.open('wb') as file:
            pass

    @staticmethod
    def load(path: Path) -> 'Vocab':
        with path.open('rb') as file:
            pass

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


if __name__ == '__main__':
    main()
