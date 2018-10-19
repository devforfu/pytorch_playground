from pathlib import Path

from fastai.data import DataBunch
from fastai.text import RNNLearner
from fastai.train import TrainingPhase, annealing_cos
from fastai.callbacks import GeneralScheduler
from fastai.text.data import TextLMDataBunch

from dataset import IMDB
from tokenizer import Vocab, tokenize_in_parallel
from utils import is_empty


DATA_ROOT = Path.home() / 'data'
IMDB_PATH = DATA_ROOT / 'aclImdb'
LM_PATH = IMDB_PATH / 'lm'
TRAIN_PATH = DATA_ROOT / 'train'
TEST_PATH = DATA_ROOT / 'test'


def main():
    datasets = create_or_restore(DATA_ROOT)
    lm_data = [datasets['train_unsup'], datasets['test_unsup']]
    bunch = TextLMDataBunch.create(lm_data, path=LM_PATH)

    lr = 1e-3
    wd = 1e-7
    cycle_len = 1
    n_cycles = 3
    cycle_mult = 2
    n = sum(len(ds) for ds in lm_data)

    phases = [
        TrainingPhase(
            length=n*(cycle_len*cycle_mult),
            lrs=lr, moms=0.9, lr_anneal=annealing_cos)
        for _ in range(n_cycles)]

    learner = RNNLearner.language_model(bunch)
    sched = GeneralScheduler(learner, phases)
    learner.fit(1, lr/2, wd=wd, callbacks=[sched])

    print('Saving model')
    learner.save('lm')
    learner.save_encoder('lm_enc')



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
