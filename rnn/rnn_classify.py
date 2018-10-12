import pickle
from textwrap import wrap
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import spacy
from spacy.symbols import ORTH

from rules import default_rules


IMDB = Path.home() / 'data' / 'aclImdb'
TRAIN_PATH = IMDB / 'train'
TEST_PATH = IMDB / 'test'
CLASSES = ['neg', 'pos', 'unsup']

BOS, FLD, UNK, PAD = SPECIAL_TOKENS = 'xxbox', 'xxfld', 'xxunk', 'xxpad'


def main():
    datasets = create_or_restore(IMDB)
    dataset = datasets['train_unsup']
    sample = dataset.vocab.textify(dataset[0])
    print(wrap(sample, width=80))


def create_or_restore(path: Path):
    """Prepared IMDB datasets from raw files, or loads previously saved objects
    into memory.
    """
    datasets_dir = path / 'datasets'

    if datasets_dir.exists():
        print('Loading data from %s' % datasets_dir)
        datasets = {}
        for filename in datasets_dir.glob('*.pickle'):
            datasets[filename.stem] = ImdbDataset.load(filename)

    else:
        print('Creating folder %s' % datasets_dir)
        datasets_dir.mkdir(parents=True)

        print('Preparing datasets...')

        train_sup = ImdbDataset(
            IMDB, supervised=True, train=True,
            tokenizer=tokenize_in_parallel,
            make_vocab=Vocab.make_vocab)

        test_sup = ImdbDataset(
            IMDB, supervised=True, train=False,
            tokenizer=tokenize_in_parallel,
            vocab=train_sup.vocab)

        train_unsup = ImdbDataset(
            IMDB, supervised=False, train=True,
            tokenizer=tokenize_in_parallel,
            make_vocab=Vocab.make_vocab)

        test_unsup = ImdbDataset(
            IMDB, supervised=False, train=False,
            tokenizer=tokenize_in_parallel,
            vocab=train_unsup.vocab)

        datasets = {
            'train_sup': train_sup,
            'test_sup': test_sup,
            'train_unsup': train_unsup,
            'test_unsup': test_unsup
        }

        for name, dataset in datasets.items():
            print(f'Saving dataset {name}')
            dataset.save(datasets_dir / f'{name}.pickle')

    for name, dataset in datasets.items():
        print(f'{name} vocab size: {dataset.vocab.size}')

    return datasets


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
    def __init__(self, root: Path, train=True, supervised=False,
                 tokenizer=None, vocab=None, make_vocab=None):
        """
        Args:
             root: Path to the folder with train and tests subfolders.
             supervised: If True, then the data from supervised subset is loaded.
             train: If True, then the data from training subset is loaded.
             vocab: Dataset vocab used to convert tokens into digits.
             make_vocab: Callable creating vocab from tokens. Note that this
                parameter should be provided in case if `vocab` doesn't present.

        """
        assert vocab or make_vocab, 'Nor vocabulary, not function provided'

        self.root = root
        self.train = train
        self.supervised = supervised

        subfolder = root / ('train' if train else 'test')
        if tokenizer is None:
            tokenizer = lambda x: x

        if supervised:
            texts, labels = [], []
            for index, label in enumerate(CLASSES):
                if label == 'unsup':
                    continue
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
                files_folder = subfolder/label
                for filename in files_folder.glob('*.txt'):
                    texts.append(filename.open('r').read())

        tokens = tokenizer(texts)
        if make_vocab:
            vocab = make_vocab(tokens)
        num_tokens = vocab.numericalize(tokens)

        self.vocab = vocab
        if train:
            self.train_data = num_tokens
        else:
            self.test_data = num_tokens

    def __getitem__(self, index):
        if self.train and self.supervised:
            return self.train_data[index], self.train_labels[index]
        elif self.train and not self.supervised:
            return self.train_data[index]
        elif not self.train and self.supervised:
            return self.test_data[index], self.test_labels[index]
        else:
            return self.test_data[index]

    def __len__(self):
        return len(self.train_data if self.train else self.test_data)

    def save(self, path):
        with path.open('wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with path.open('rb') as file:
            dataset = pickle.load(file)
        return dataset


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




if __name__ == '__main__':
    main()
