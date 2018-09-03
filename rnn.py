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


class Dataset:

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
    def __init__(self, seq, bptt=10, batch_size=64, random_length=True):
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
                 architecture=nn.RNN, device=DEVICE):

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.device = device

        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_factors)
        self.rnn = architecture(n_factors, n_hidden)
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
            h = torch.zeros(2, 1, batch_size, self.n_hidden)
        else:
            h = torch.zeros(1, batch_size, self.n_hidden)
        return h.to(self.device)


def truncate_history(v):
    if type(v) == torch.Tensor:
        return v.detach()
    else:
        return tuple(truncate_history(x) for x in v)


class StringBuilder:
    """
    The helper class used during debugging process to convert tensors with
    integer indexes into strings with batches of text they represent.
    """
    def __init__(self, field):
        self.field = field

    def __call__(self, tensor):
        return '\n'.join([
            ''.join([
                self.field.vocab.itos[char]
                for char in line])
            for line in tensor])


class Stepper:

    def __init__(self, model, optimizer, schedule, loss):
        schedule.step()
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.loss = loss

    def step(self, x, y, train=True):
        with torch.set_grad_enabled(train):
            out = self.model(x)
            loss = self.loss(out, y.view(-1))
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
        return loss.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


class Callback:

    def training_start(self):
        pass

    def training_end(self):
        pass

    def epoch_start(self, epoch, phase):
        pass

    def epoch_end(self, epoch, phase):
        pass

    def batch_start(self, epoch, phase):
        pass

    def batch_end(self, epoch, phase):
        pass


class CallbackGroup(Callback):

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def training_start(self):
        for cb in self.callbacks: cb.training_start()

    def training_end(self):
        for cb in self.callbacks: cb.training_end()

    def epoch_start(self, epoch, phase):
        for cb in self.callbacks: cb.epoch_start(epoch, phase)

    def epoch_end(self, epoch, phase):
        for cb in self.callbacks: cb.epoch_end(epoch, phase)

    def batch_start(self, epoch, phase):
        for cb in self.callbacks: cb.batch_start(epoch, phase)

    def batch_end(self, epoch, phase):
        for cb in self.callbacks: cb.batch_start(epoch, phase)

    def set_loop(self, loop):
        for cb in self.callbacks: cb.loop = loop


class Logger(Callback):

    def __init__(self, streams=None):
        self.streams = streams or [sys.stdout]
        self.epoch_history = {}
        self.curr_epoch = 0

    def epoch_end(self, epoch, phase):
        if self.curr_epoch != epoch:
            metrics = ' '.join([
                f'{name: >5s} - {loss:2.4f}'
                for name, loss in self.epoch_history.items()])
            string = f'Epoch {epoch:4d}: {metrics}\n'
            for stream in self.streams:
                stream.write(string)
                stream.flush()
            self.curr_epoch = epoch

        self.epoch_history[phase.name] = phase.avg_loss


class EarlyStopping(Callback):

    def __init__(self, patience=3, phase='valid', metric='avg_loss',
                 folder=None, better=min, save_model=True):

        self.patience = patience
        self.phase = phase
        self.metric = metric
        self.folder = folder or os.getcwd()
        self.better = better
        self.save_model = save_model
        self.no_improvement = None
        self.best_value = None
        self.best_model = None
        self.stopped_on_epoch = None
        self.loop = None

    def set_loop(self, loop):
        self.loop = loop

    def training_start(self):
        assert self.loop is not None
        self.no_improvement = 0

    def epoch_end(self, epoch, phase):
        if phase.name != self.phase:
            return

        value = getattr(phase, self.metric)
        if not value:
            return

        best_value = self.best_value or value
        better = self.better(best_value, value) == value
        if not better:
            self.no_improvement += 1
        else:
            path = f'model_{self.phase}_{self.metric}_{value:2.4f}.weights'
            self.best_value = value
            self.no_improvement = 0
            if self.save_model:
                best_model = join(self.folder, path)
                self.loop.save_model(best_model)
                self.best_model = best_model

        if self.no_improvement >= self.patience:
            self.loop.stop = True
            self.stopped_on_epoch = epoch


class Phase:

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.batch_num = 0
        self.avg_loss = 0

    def __repr__(self):
        return f'<Phase: {self.name}, avg_loss: {self.avg_loss:2.4f}>'


class Loop:

    def __init__(self, stepper, alpha=0.98):
        self.stepper = stepper
        self.alpha = alpha
        self.stop = False

    def run(self, train_data, valid_data, epochs=100, callbacks=None):
        phases = [
            Phase(name='train', dataset=train_data),
            Phase(name='valid', dataset=valid_data)
        ]

        cb = CallbackGroup(callbacks)
        cb.set_loop(self)
        cb.training_start()

        a = self.alpha
        for epoch in range(epochs):
            if self.stop:
                break
            for phase in phases:
                cb.epoch_start(epoch, phase)
                is_training = phase.name == 'train'
                for x, y in phase.dataset:
                    phase.batch_num += 1
                    cb.batch_start(epoch, phase)
                    loss = self.stepper.step(x, y, is_training)
                    phase.avg_loss = phase.avg_loss*a + loss*(1 - a)
                    cb.batch_end(epoch, phase)
                cb.epoch_end(epoch, phase)
        cb.training_end()

    def save_model(self, path):
        self.stepper.save_model(path)


def generate_text(model, field, seed, n=500):
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
    n_epochs = 100

    field = Field(lower=True, tokenize=list)
    dataset = Dataset(field, min_freq=5)
    factory = lambda text: SequenceIterator(text, bptt, bs)
    dataset.build(TRAIN_PATH, VALID_PATH, factory)

    model = RNN(dataset.vocab_size, n_factors, bs, n_hidden, architecture=nn.LSTM)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    sched = CosineAnnealingLR(optimizer, t_max=dataset['train'].total_iters)

    early_stopping, logger = EarlyStopping(patience=10), Logger()
    stepper = Stepper(model, optimizer, sched, F.nll_loss)
    loop = Loop(stepper)

    loop.run(train_data=dataset['train'],
             valid_data=dataset['valid'],
             epochs=n_epochs,
             callbacks=[early_stopping, logger])

    model.load_state_dict(torch.load(early_stopping.best_model))
    text = generate_text(model, field, seed='For thos')
    pretty_print(text)


if __name__ == '__main__':
    main()
