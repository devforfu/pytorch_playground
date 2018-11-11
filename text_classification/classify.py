from pathlib import Path

from fastai.train import TrainingPhase, annealing_cos
from fastai.text import TextDataset, TextLMDataBunch, RNNLearner
from fastai.callbacks import GeneralScheduler
from fastai.callbacks.tracker import EarlyStopping, SaveModel


DATA_ROOT = Path.home()/'data'
IMDB_PATH = DATA_ROOT/'aclImdb'
LM_PATH = IMDB_PATH/'lm'


def main():
    train_ds = TextDataset.from_folder(IMDB_PATH, name='train', shuffle=True)
    valid_ds = TextDataset.from_folder(IMDB_PATH, name='test')
    lm_data = [train_ds, valid_ds]
    lm_bunch = TextLMDataBunch.create(lm_data, path=LM_PATH)

    learner = RNNLearner.language_model(lm_bunch)

    n = sum(len(ds) for ds in lm_data)
    num_epochs, phases = create_phases(3, n)

    callbacks = [
        EarlyStopping(learner, patience=2),
        SaveModel(learner),
        GeneralScheduler(learner, phases)
    ]

    learner.fit(num_epochs, )


def create_phases(n_cycles, n_items, cycle_mult=2, cycle_len=1, mom=0.8,
                  lr=1e-3):

    phases = [
        TrainingPhase(
            n_items*(cycle_len * cycle_mult**i),
            lrs=lr, moms=mom, lr_anneal=annealing_cos
        ) for i in range(n_cycles)]

    if cycle_mult == 1:
        total_epochs = n_cycles * cycle_len
    else:
        total_epochs = int(
            cycle_len*(1 - cycle_mult**n_cycles)/(1 - cycle_mult))

    return total_epochs, phases



if __name__ == '__main__':
    main()
