from pathlib import Path
import pickle
import tarfile
from typing import Callable

import requests
from torch.utils.data import Dataset

from utils import is_empty


class IMDB(Dataset):
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
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    archive_size = 84125825
    classes = ('pos', 'neg', 'unsup')

    def __init__(self, root: Path, train: bool=True, supervised: bool=False,
                 tokenizer=None, vocab=None, make_vocab: Callable=None,
                 download: bool=True):
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

        subfolder = root / 'aclImdb' / ('train' if train else 'test')
        if is_empty(subfolder):
            if not download:
                raise FileNotFoundError(
                    'Required files not found! Check if folder with IMDB data exists')
            self.download(root)

        if tokenizer is None:
            tokenizer = identity

        if supervised:
            texts, labels = [], []
            for index, label in enumerate(self.classes):
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
            for label in self.classes:
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

    def save(self, path: Path):
        with path.open('wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: Path):
        with path.open('rb') as file:
            dataset = pickle.load(file)
        return dataset

    @staticmethod
    def download(path: Path):
        archive = path / 'imdb.tag.gz'

        if not archive.exists():
            req = requests.get(IMDB.url)
            req.raise_for_status()
            assert len(req.content) == IMDB.archive_size, 'Downloading failure!'
            with archive.open('wb') as file:
                file.write(req.content)

        with tarfile.open(archive) as arch:
            arch.extractall(path)


def identity(x):
    return x
