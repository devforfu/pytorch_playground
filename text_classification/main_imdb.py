from pathlib import Path

import fire
from fastai.data import DataBunch
from fastai.text import RNNLearner
from fastai.train import TrainingPhase, annealing_cos
from fastai.callbacks import GeneralScheduler
from fastai.callbacks.tracker import EarlyStopping, SaveModel
from fastai.text.data import TextLMDataBunch
from torch.nn import functional as F

from dataset import IMDB
from tokenizer import Vocab, tokenize_in_parallel
from utils import is_empty


DATA_ROOT = Path.home() / 'data'
IMDB_PATH = DATA_ROOT / 'aclImdb'
LM_PATH = IMDB_PATH / 'lm'
TRAIN_PATH = DATA_ROOT / 'train'
TEST_PATH = DATA_ROOT / 'test'


def main():
    fire.Fire(train_lm)


def train_lm(n_cycles: int=3, cycle_len: int=1, cycle_mult: int=2,
             momentum: float=0.8, bptt: int=40, lr: float=1e-3,
             wd: float=1e-7):

    datasets = create_or_restore(DATA_ROOT)
    lm_data = [
        fastai_patch(ds) for ds in (
            datasets['train_unsup'], datasets['test_unsup'])]
    bunch = TextLMDataBunch.create(lm_data, path=LM_PATH, bptt=bptt)

    n = sum(len(ds) for ds in lm_data)
    phases = [
        TrainingPhase(
            n*(cycle_len * cycle_mult**i),
            lrs=lr, moms=momentum, lr_anneal=annealing_cos
        ) for i in range(n_cycles)]
    learner = RNNLearner.language_model(bunch, bptt)
    cbs = [
        EarlyStopping(learner, patience=2),
        GeneralScheduler(learner, phases),
        SaveModel(learner)]

    if cycle_mult == 1:
        total_epochs = n_cycles * cycle_len
    else:
        total_epochs = int(cycle_len*(1 - cycle_mult**n_cycles)/(1 - cycle_mult))

    print(f'Total number of epochs: {total_epochs:d}')
    try:
        learner.fit(total_epochs, wd=wd, callbacks=cbs)
    except RuntimeError as e:
        print(f'Model training error: {e}')
    finally:
        folder = learner.path/learner.model_dir
        print(f'Saving latest model state into {folder}')
        learner.save('lm_final')
        learner.save_encoder('lm_final_enc')


def fastai_patch(ds):
    """Adding properties to the dataset required to be compatible with
    fastai library.
    """
    ds.__dict__['ids'] = ds.train_data if ds.train else ds.test_data
    ds.__dict__['vocab_size'] = ds.vocab.size
    ds.__dict__['loss_func'] = F.cross_entropy
    return ds


def create_or_restore(path: Path):
    """Prepared IMDB datasets from raw files, or loads previously saved objects
    into memory.
    """
    datasets_dir = path / 'aclImdb' / 'datasets'

    if datasets_dir.exists() and not is_empty(datasets_dir):
        print('Loading data from %s' % datasets_dir)
        datasets = {}
        for filename in datasets_dir.glob('*.pickle'):
            datasets[filename.stem] = IMDB.load(filename)

    else:
        print('Creating folder %s' % datasets_dir)

        datasets_dir.mkdir(parents=True, exist_ok=True)

        print('Tokenizing supervised data (sentiment classification)')

        train_sup = IMDB(
            path, supervised=True, train=True,
            tokenizer=tokenize_in_parallel,
            make_vocab=Vocab.make_vocab)

        test_sup = IMDB(
            path, supervised=True, train=False,
            tokenizer=tokenize_in_parallel,
            vocab=train_sup.vocab)

        print('Tokenizing unsupervised data (language model)')

        train_unsup = IMDB(
            path, supervised=False, train=True,
            tokenizer=tokenize_in_parallel,
            make_vocab=Vocab.make_vocab)

        test_unsup = IMDB(
            path, supervised=False, train=False,
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


if __name__ == '__main__':
    main()
