import pickle
import argparse
from textwrap import wrap
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np

import spacy
from spacy.symbols import ORTH

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset

from rules import default_rules
from core.loop import Loop
from core.metrics import accuracy
from core.schedule import CosineAnnealingLR
from core.callbacks import default_callbacks


IMDB = Path.home() / 'data' / 'aclImdb'
TRAIN_PATH = IMDB / 'train'
TEST_PATH = IMDB / 'test'
CLASSES = ['neg', 'pos', 'unsup']

BOS, FLD, UNK, PAD = SPECIAL_TOKENS = 'xxbox', 'xxfld', 'xxunk', 'xxpad'


def main():
    datasets = create_or_restore(IMDB)
    train_data = datasets['train_unsup']
    test_data = datasets['test_unsup']

    bs = 50
    bptt = 70

    train = SequenceIterator(to_sequence(train_data), bptt, bs)
    valid = SequenceIterator(to_sequence(test_data), bptt, bs)

    lm = LanguageModel(
        vocab_sz=train_data.vocab.size,
        embed_sz=400, n_hidden=1150)

    dev = device(force_cpu=True) if args.use_cpu else device(args.cuda)
    print('Selected device: %s' % dev)

    opt = optim.Adam(
        lm.parameters(), lr=1e-3, weight_decay=1e-7, betas=(0.8, 0.99))
    cycle_length = len(train_data) // bs
    sched = CosineAnnealingLR(opt, t_max=cycle_length, cycle_mult=1, eta_min=1e-5)
    loop = Loop(lm, opt, sched, device=dev)

    loop.run(train_data=train, valid_data=valid,
             loss_fn=F.cross_entropy,
             metrics=[accuracy],
             callbacks=default_callbacks())

    best_model = loop['Checkpoint'].best_model
    print('Best model: %s' % best_model)
    with open('best', 'w') as file:
        file.write(best_model + '\n')


def parse_args():
    argparse.ArgumentParser()


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


class ImdbDataset(Dataset):
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


def compact_print(string):
    print('\n'.join(wrap(string, width=80)))



class SequenceIterator:
    """A wrapper on top of IMDB dataset that converts numericalized
    observations into format, suitable to train a language model.

    To train a language model, one needs to convert an unsupervised dataset
    into two 2D arrays with tokens. The first array contains "previous" words,
    and the second one - "next" words. Each "previous" word is used to predict
    the "next" one. Therefore, we're getting a supervised training task.
    """
    def __init__(self, seq, bptt=10, split_size=64, random_length=True,
                 flatten_target=True):

        n_batches = seq.shape[0] // split_size
        truncated = seq[:n_batches * split_size]
        batches = truncated.view(split_size, -1).t().contiguous()

        self.bptt = bptt
        self.split_size = split_size
        self.random_length = random_length
        self.flatten_target = flatten_target
        self.batches = batches
        self.curr_iter = 0
        self.curr_line = 0
        self.total_lines = batches.shape[0]
        self.total_iters = self.total_lines // self.bptt - 1

    @property
    def completed(self):
        if self.curr_line >= self.total_lines - 1:
            return True
        if self.curr_iter >= self.total_iters:
            return True
        return False

    def __iter__(self):
        self.curr_line = self.curr_iter = 0
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.completed:
            raise StopIteration()
        seq_len = self.get_sequence_length()
        batch = self.get_batch(seq_len)
        self.curr_line += seq_len
        self.curr_iter += 1
        return batch

    def get_sequence_length(self):
        """
        Returns a length of sequence taken from the dataset to form a batch.

        By default, this value is based on the value of bptt parameter but
        randomized during training process to pick sequences of characters with
        a bit different length.
        """
        if self.random_length is None:
            return self.bptt
        bptt = self.bptt
        if np.random.random() >= 0.95:
            bptt /= 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        return seq_len

    def get_batch(self, seq_len):
        """
        Picks training and target batches from the source depending on current
        iteration number.
        """
        i, source = self.curr_line, self.batches
        seq_len = min(seq_len, self.total_lines - 1 - i)
        X = source[i:i + seq_len].contiguous()
        y = source[(i + 1):(i + 1) + seq_len].contiguous()
        if self.flatten_target:
            y = y.view(-1)
        return X, y


def to_sequence(dataset):
    seq = concat(dataset.train_data if dataset.train else dataset.test_data)
    return torch.LongTensor(seq)


def concat(arrays):
    seq = []
    dtype = arrays[0].dtype
    for arr in arrays:
        seq.extend(arr.tolist())
    return np.array(seq, dtype=dtype)


def to_np(tensor):
    return tensor.detach().cpu().numpy()


class RNNCore(nn.Module):

    init_range = 0.1

    def __init__(self, vocab_sz: int, embed_sz: int, n_hidden: int,
                 n_layers: int, pad_idx: int):

        def get_size(index):
            """Returns RNN cell input and hidden size depending on its position
            in the network.
            """
            if index == 0:
                return embed_sz, n_hidden
            elif index == n_layers - 1:
                return n_hidden, embed_sz
            return n_hidden, n_hidden


        def create_lstm():
            return [nn.LSTM(*get_size(l), 1) for l in range(n_layers)]


        super().__init__()
        self.encoder = nn.Embedding(vocab_sz, embed_sz, padding_idx=pad_idx)
        self.rnns = nn.ModuleList(create_lstm())

        self.hidden_sizes = [layer.hidden_size for layer in self.rnns]
        self.embed_sz = embed_sz
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bs = None
        self.hidden = None
        self.weights = None
        self._init()

    @property
    def output_size(self):
        return self.hidden_sizes[-1]

    def forward(self, tensor):
        seq_len, bs = tensor.size()
        if bs != self.bs:
            self.bs = bs
            self.create_hidden()

        raw_output = self.encoder(tensor)
        raw_outputs, new_hidden = [], []
        for index, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, self.hidden[index])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
        self.hidden = truncate_history(new_hidden)
        return raw_outputs

    def reset(self):
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]

    def create_hidden(self):
        self.reset()
        self.weights = next(self.parameters()).data
        self.hidden = [
            (self._hidden(sz), self._hidden(sz))
            for sz in self.hidden_sizes]

    def _hidden(self, sz):
        return self.weights.new(1, self.bs, sz).zero_()

    def _init(self):
        a = self.init_range
        self.encoder.weight.data.uniform_(-a, a)


class WeightDropout(nn.Module):

    def __init__(self, module: nn.Module, weight_p: float,
                 layer_names=('weight_hh_10')):

        super().__init__()
        self.module = module
        self.weight_p = weight_p
        self.layer_names = layer_names

        for layer in self.layer_names:
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))

    def forward(self, *tensors):
        self._set_weights()
        return self.module.forward(*tensors)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(
                raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'):
            self.module.reset()

    def _set_weights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(
                raw_w, p=self.weight_p, training=self.training)


class LanguageModel(nn.Module):
    """A RNN-based model predicting next word from the previous one."""

    init_range = 0.1

    def __init__(self, vocab_sz: int, embed_sz: int, n_hidden: int=1000,
                 n_layers: int=3, bias: bool=True, padding_idx=1):

        super().__init__()
        self.rnn = RNNCore(vocab_sz, embed_sz, n_hidden, n_layers, padding_idx)
        self.decoder = nn.Linear(self.rnn.output_size, vocab_sz, bias=bias)
        self._init(bias)

    def forward(self, tensor):
        raw_outputs = self.rnn.forward(tensor)
        last = raw_outputs[-1]
        input_shape = last.size(0)*last.size(1), last.size(2)
        decoded = self.decoder(last.view(input_shape))
        return decoded

    def _init(self, bias):
        a = self.init_range
        self.decoder.weight.data.uniform_(-a, a)
        if bias:
            self.decoder.bias.data.zero_()


def truncate_history(v):
    """
    Detaches tensor from its computational history.
    """
    if type(v) == torch.Tensor:
        return v.detach()
    else:
        return tuple(truncate_history(x) for x in v)


def device(i=0, force_cpu=True):
    name = f'cuda:{i}' if torch.cuda.is_available() else 'cpu'
    if force_cpu:
        name = 'cpu'
    return torch.device(name)


if __name__ == '__main__':
    main()
