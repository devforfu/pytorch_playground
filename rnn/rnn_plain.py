import os
import sys
import math
import textwrap
from os.path import expanduser, join

import numpy as np

import torch
from torch import nn
from torch import optim
from torchtext.data import Field
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler


PATH = expanduser(join('~', 'data', 'fastai', 'nietzsche'))
TRAIN_PATH = join(PATH, 'trn', 'train.txt')
VALID_PATH = join(PATH, 'val', 'valid.txt')


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def create_dataset(bptt, batch_size):
    field = Field(lower=True, tokenize=list)
    dataset = Dataset(field, min_freq=5)
    factory = lambda seq: SequenceIterator(seq, bptt, batch_size)
    dataset.build(TRAIN_PATH, VALID_PATH, factory)
    return dataset, field


class Dataset:
    """
    Represents a set of encoded texts prepared for model training and
    validation.
    """
    def __init__(self, field: Field, min_freq: int=1):
        self.field = field
        self.min_freq = min_freq
        self.subsets = {}
        self.vocab_size = None

    def build(self, train: str, valid: str, iterator_factory):
        content_per_file = {}
        for name, path in [('train', train), ('valid', valid)]:
            file_content = []
            new_line = False
            with open(path) as file:
                for line in file:
                    if line.endswith('\n'):
                        new_line = True
                        if line == '\n':
                            continue
                    file_content += self.field.preprocess(line)
                    if new_line:
                        file_content.append(' ')
                        new_line = False
            content_per_file[name] = file_content

        train_text = content_per_file['train']
        self.field.build_vocab(train_text, min_freq=self.min_freq)
        self.vocab_size = len(self.field.vocab.itos)

        for name, content in content_per_file.items():
            sequence = self.field.numericalize(content)
            iterator = iterator_factory(sequence.view(-1))
            self.subsets[name] = iterator

    def __getitem__(self, item):
        if item not in self.subsets:
            raise ValueError(f'Unexpected dataset name: {item}')
        return self.subsets[item]


class SequenceIterator:
    """
    A simple iterator to convert an array of encoded characters into group of
    batches reshaped into format, appropriate for the RNN training process.
    """
    def __init__(self, seq, bptt=10, batch_size=64, random_length=True,
                 flatten_target=True):

        # Converting dataset into batches:
        # 1) truncate text length to evenly fit into number of batches
        # 2) reshape the text into N (# of batches) * M (batch size)
        # 3) transpose to convert into "long" format with fixed number of cols

        n_batches = seq.size(0) // batch_size
        truncated = seq[:n_batches * batch_size]
        batches = truncated.view(batch_size, -1).t().contiguous()

        self.bptt = bptt
        self.batch_size = batch_size
        self.random_length = random_length
        self.flatten_target = flatten_target
        self.batches = batches
        self.curr_line = 0
        self.curr_iter = 0
        self.total_lines = batches.size(0)
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


class CosineAnnealingLR(_LRScheduler):
    """
    A scheduler implementing cosine annealing with restarts and an increasing
    period of the decay.
    """
    def __init__(self, optimizer, t_max=200, eta_min=0.0005,
                 cycle_mult=2, last_epoch=-1):

        self.t_max = t_max
        self.eta_min = eta_min
        self.cycle_mult = cycle_mult
        self.cycle_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        self.cycle_epoch += 1

        t_max = self.t_max
        eta_min = self.eta_min
        t = self.cycle_epoch % t_max

        learning_rates = []
        for lr in self.base_lrs:
            delta = lr - eta_min
            new_lr = eta_min + delta*(1 + math.cos(math.pi * t/t_max)) / 2
            learning_rates.append(new_lr)

        if t == 0:
            self.cycle_epoch = 0
            self.t_max *= self.cycle_mult

        return learning_rates


class RNN(nn.Module):

    def __init__(self, vocab_size, n_factors, batch_size, n_hidden,
                 n_recurrent=1, architecture=nn.RNN, dropout=0.5,
                 device=DEVICE):

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_recurrent = n_recurrent
        self.device = device

        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_factors)
        self.rnn = architecture(
            n_factors, n_hidden,
            dropout=dropout, num_layers=n_recurrent)
        self.out = nn.Linear(n_hidden, vocab_size)
        self.hidden_state = self.init_hidden(batch_size).to(device)
        self.batch_size = batch_size
        self.to(device)

    def forward(self, batch):
        bs = batch.size(1)
        if bs != self.batch_size:
            self.hidden_state = self.init_hidden(bs)
            self.batch_size = bs
        embeddings = self.embed(batch)
        rnn_outputs, h = self.rnn(embeddings, self.hidden_state)
        self.hidden_state = truncate_history(h)
        linear = self.out(rnn_outputs)
        return F.log_softmax(linear, dim=-1).view(-1, self.vocab_size)

    def init_hidden(self, batch_size):
        if type(self.rnn) == nn.LSTM:
            # an LSTM cell requires two hidden states
            h = torch.zeros(2, self.n_recurrent, batch_size, self.n_hidden)
        else:
            h = torch.zeros(self.n_recurrent, batch_size, self.n_hidden)
        return h.to(self.device)


def truncate_history(v):
    """
    Detaches tensor from its computational history.
    """
    if type(v) == torch.Tensor:
        return v.detach()
    else:
        return tuple(truncate_history(x) for x in v)


def generate_text(model, field, seed, n=500):
    """
    Generates text using trained model and an initial seed.
    """
    string = seed
    for i in range(n):
        indexes = field.numericalize(string)
        predictions = model(indexes.transpose(0, 1))
        last_output = predictions[-1]
        [most_probable] = torch.multinomial(last_output.exp(), 1)
        char = field.vocab.itos[most_probable]
        seed = seed[1:] + char
        string += char
    return string


def pretty_print(text, width=80):
    print('\n'.join(textwrap.wrap(text, width=width)))


def main():
    bs = 64
    bptt = 8
    n_factors = 50
    n_hidden = 256

    dataset, field = create_dataset(bptt, bs)

    # model and optimizer initialization
    model = RNN(
        vocab_size=dataset.vocab_size,
        n_factors=n_factors,
        batch_size=bs,
        n_hidden=n_hidden,
        architecture=nn.LSTM)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    sched = CosineAnnealingLR(optimizer, t_max=dataset['train'].total_iters)

    # performance metrics
    train_avg_loss = 0
    valid_avg_loss = 0
    train_batch_num = 0
    valid_batch_num = 0
    n_epochs = 20
    alpha = 0.98
    patience = 3
    no_improvement = 0
    best_loss = np.inf
    best_model = None

    for epoch in range(1, n_epochs + 1):

        # update model's parameters
        for x, y in dataset['train']:
            train_batch_num += 1
            sched.step()
            optimizer.zero_grad()
            loss = F.nll_loss(model(x), y.view(-1))
            loss.backward()
            optimizer.step()
            train_avg_loss = train_avg_loss*alpha + loss.item()*(1 - alpha)

        # validate performance
        for x, y in dataset['valid']:
            valid_batch_num += 1
            with torch.no_grad():
                loss = F.nll_loss(model(x), y.view(-1))
                valid_avg_loss = valid_avg_loss*alpha + loss.item()*(1 - alpha)

        train_epoch_loss = train_avg_loss / (1 - alpha ** train_batch_num)
        valid_epoch_loss = valid_avg_loss / (1 - alpha ** valid_batch_num)
        print('Epoch %03d - train: %2.4f - valid: %2.4f' % (
            epoch, train_epoch_loss, valid_epoch_loss
        ))

        if valid_epoch_loss >= best_loss:
            no_improvement += 1
        else:
            filename = f'model_{valid_epoch_loss:2.4f}.weights'
            torch.save(model.state_dict(), filename)
            best_loss = valid_epoch_loss
            best_model = filename
            no_improvement = 0

        if no_improvement >= patience:
            print('Early stopping...')
            break

    print('\nGenerated text:')
    model.load_state_dict(torch.load(best_model))
    pretty_print(generate_text(model, field, 'For thos'))


if __name__ == '__main__':
    main()
