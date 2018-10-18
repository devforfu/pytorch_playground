from pathlib import Path

from fastai.data import DataBunch
from fastai.text import RNNLearner

from dataset import IMDB
from tokenizer import Vocab, tokenize_in_parallel
from utils import is_empty


DATA_ROOT = Path.home() / 'data'
TRAIN_PATH = DATA_ROOT / 'train'
TEST_PATH = DATA_ROOT / 'test'


def main():
    datasets = create_or_restore(DATA_ROOT)

    bunch = DataBunch.create(datasets['train'], datasets['test'])
    learner = RNNLearner.language_model(bunch)
    learner.fit(1)


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
