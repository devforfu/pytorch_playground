"""
The RNN training implementation using object-oriented classes hierarchy.
"""
import textwrap
from os.path import expanduser, join

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchtext.data import Field

from core.text import TextDataset
from core.loop import Loop, Stepper
from core.iterators import SequenceIterator
from core.schedule import CosineAnnealingLR
from core.callbacks import EarlyStopping, Checkpoint, Logger, History


ROOT = expanduser(join('~', 'data', 'fastai', 'nietzsche'))
TRAIN_DIR = join(ROOT, 'trn')
VALID_DIR = join(ROOT, 'val')


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class RNN(nn.Module):

    def __init__(self, vocab_size, n_factors, batch_size, n_hidden,
                 n_recurrent=1, architecture=nn.RNN, device=DEVICE):

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_recurrent = n_recurrent
        self.device = device

        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_factors)
        self.rnn = architecture(n_factors, n_hidden, num_layers=n_recurrent)
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
    if type(v) == torch.Tensor:
        return v.detach()
    else:
        return tuple(truncate_history(x) for x in v)


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
    n_recurrent = 1
    n_epochs = 100

    field = Field(lower=True, tokenize=list)
    dataset = TextDataset(field, min_freq=5)
    factory = lambda seq: SequenceIterator(seq, bptt, bs)
    dataset.build(train=TRAIN_DIR, valid=VALID_DIR, iterator_factory=factory)

    model = RNN(dataset.vocab_size,
                n_factors=n_factors,
                batch_size=bs,
                n_hidden=n_hidden,
                n_recurrent=n_recurrent,
                architecture=nn.LSTM)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    cycle_length = dataset['train'].total_iters
    scheduler = CosineAnnealingLR(optimizer, t_max=cycle_length)
    loop = Loop(Stepper(model, optimizer, scheduler, F.nll_loss))

    loop.run(train_data=dataset['train'],
             valid_data=dataset['valid'],
             epochs=n_epochs,
             callbacks=[
                 EarlyStopping(patience=3),
                 Logger(),
                 History(),
                 Checkpoint()])

    best_model = loop['Checkpoint'].best_model
    model.load_state_dict(torch.load(best_model))
    text = generate_text(model, field, seed='For thos')
    pretty_print(text)


if __name__ == '__main__':
    main()