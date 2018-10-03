"""
Training RNN on lyrics downloaded from AZLyrics website.
"""
from os.path import expanduser, join

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchtext.data import Field

from rnn_plain import RNN, generate_text, pretty_print
from core.text import TextDataset
from core.loop import Loop, Stepper
from core.iterators import SequenceIterator
from core.schedule import CosineAnnealingLR
from core.callbacks import EarlyStopping, Checkpoint, Logger, History


ROOT = expanduser(join('~', 'data', 'azlyrics', 'lyrics'))
TRAIN_DIR = join(ROOT, 'train')
VALID_DIR = join(ROOT, 'valid')


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
    bs, bptt = 32, 16
    field = Field(lower=True, tokenize=list)
    dataset = TextDataset(field, keep_new_lines=True, min_freq=5)
    factory = lambda seq: SequenceIterator(seq, bptt, bs)
    dataset.build(train=TRAIN_DIR, valid=VALID_DIR, iterator_factory=factory)

    model = RNN(dataset.vocab_size,
                n_factors=50,
                batch_size=128,
                n_hidden=256,
                n_recurrent=3,
                architecture=nn.LSTM)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    cycle_length = dataset['train'].total_iters
    scheduler = CosineAnnealingLR(optimizer, t_max=cycle_length/2, eta_min=1e-5)
    callbacks = [EarlyStopping(patience=50), Logger(), History(), Checkpoint()]
    loop = Loop(Stepper(model, optimizer, scheduler, F.nll_loss))

    loop.run(train_data=dataset['train'],
             valid_data=dataset['valid'],
             callbacks=callbacks,
             epochs=500)

    best_model = loop['Checkpoint'].best_model
    model.load_state_dict(torch.load(best_model))
    text = generate_text(model, field, seed='Deep song')
    pretty_print(text)


if __name__ == '__main__':
    main()
