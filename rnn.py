from os.path import expanduser, join

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchtext import vocab, data

from fastai.nlp import LanguageModelData
from fastai.model import fit
from fastai.lm_rnn import repackage_var
from fastai.core import V


PATH = expanduser(join('~', 'data', 'fastai', 'nietzsche'))
TRAIN_PATH = join(PATH, 'trn')
VALID_PATH = join(PATH, 'val')


n_hidden = 256


class CharSeqStatefulRnn(nn.Module):

    def __init__(self, vocab_size, n_fac, bs):
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size

        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.h = self.init_hidden(bs)

    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h.size(1) != bs:
            self.init_hidden(bs)
        outp, h = self.rnn(self.e(cs), self.h)
        self.h = repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)

    def init_hidden(self, bs):
        return V(torch.zeros(1, bs, self.n_hidden))


def prepare_dataset(filename):
    text = []
    field = data.Field(lower=True, tokenize=list)
    with open(filename) as file:
        for line in file:
            text += field.preprocess(line)
    text += '<eos>'
    field.build_vocab(text, min_freq=3)
    indexes = field.numericalize(text)
    return field, indexes.view(-1)


class SequenceIterator:

    def __init__(self, seq, bptt=10, batch_size=64, random_length=True):
        n_batches = seq.size(0) // batch_size
        truncated = seq[:n_batches * batch_size]
        batches = truncated.view(batch_size, -1)

        self.bptt = bptt
        self.batch_size = batch_size
        self.random_length = random_length
        self.batches = batches
        self.curr_line = 0
        self.curr_iter = 0
        self.total_lines = batches.size(-1)
        self.total_iters = self.total_lines // self.bptt - 1

    @property
    def completed(self):
        if self.curr_line >= self.total_lines:
            return True
        if self.curr_iter >= self.total_iters:
            return True
        return False

    def __iter__(self):
        self.index = self.current = 0
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
        seq_len = self.bptt
        if self.random_length is not None:
            bptt = self.bptt
            if np.random.random() >= 0.95:
                bptt /= 2
            seq_len = max(5, int(np.random.normal(bptt, 5)))
        return seq_len

    def get_batch(self, seq_len):
        i, source = self.curr_line, self.batches
        seq_len = min(seq_len, self.total_lines - 1 - i)
        X = source[:,       i:      i + seq_len].contiguous()
        y = source[:, (i + 1):(i + 1) + seq_len].contiguous()
        return X, y.view(-1)


def main():
    field, indexes = prepare_dataset(join(TRAIN_PATH, 'train.txt'))
    iterator = SequenceIterator(indexes[:1024], bptt=10, batch_size=20)
    for x, y in iterator:
        print(x)
        print(y)
    print('Done')

    # print(indexes)

    text = data.Field(lower=True, tokenize=list)
    bs = 64
    bptt = 8
    n_factors = 42
    files = dict(train=TRAIN_PATH, validation=VALID_PATH, test=VALID_PATH)
    md = LanguageModelData.from_text_files(PATH, text, **files,
                                           bs=bs, bptt=bptt, min_freq=3)
    print(len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text))
    m = CharSeqStatefulRnn(md.nt, n_factors, bs).cuda()
    opt = optim.Adam(m.parameters(), 1e-3)
    fit(m, md, 4, opt, F.nll_loss)


if __name__ == '__main__':
    main()
