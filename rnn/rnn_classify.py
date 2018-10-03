import re
import html
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fastai.text import Tokenizer, partition_by_cores


BOS = 'xbox'   # beginning-of-sentence tag
FLD = 'xfld'  # field-data tag

DATA_ROOT = Path.home() / 'data'
IMDB = DATA_ROOT / 'aclImdb'
SENTINEL = IMDB / 'created'
CLASS_PATH = DATA_ROOT / 'imdb_class'
LM_PATH =  DATA_ROOT / 'imdb_lm'
CLASSES = ['neg', 'pos', 'unsup']

RE_SPACE = re.compile(r'  +')

RANDOM_SEED = 42


def main():
    np.random.seed(RANDOM_SEED)

    if not SENTINEL.exists():
        create_if_not_exist(CLASS_PATH, LM_PATH)
        dataset = create_dataset(IMDB)
        save_data(dataset, CLASS_PATH, supervised=True)
        save_data(dataset, LM_PATH, supervised=False)
        SENTINEL.open('w').write(1)

    cls_data, lm_data = [load_data(path) for path in (CLASS_PATH, LM_PATH)]


def get_texts(path):
    texts, labels = [], []
    for index, label in enumerate(CLASSES):
        for filename in (path/label).glob('*.*'):
            texts.append(filename.open('r').read())
            labels.append(index)
    return np.array(texts), np.array(labels)


def create_if_not_exist(*folders):
    for path in folders:
        if not path.exists():
            path.mkdir(parents=True)


def create_dataset(root):
    trn_texts, trn_labels = get_texts(root / 'train')
    val_texts, val_labels = get_texts(root / 'test')

    trn_idx = np.random.permutation(len(trn_texts))
    val_idx = np.random.permutation(len(val_texts))

    trn_texts = trn_texts[trn_idx]
    val_texts = val_texts[val_idx]

    trn_labels = trn_labels[trn_idx]
    val_labels = val_labels[val_idx]

    col_names = ['labels', 'text']
    df_trn = pd.DataFrame({'text': trn_texts, 'labels': trn_labels}, columns=col_names)
    df_val = pd.DataFrame({'text': val_texts, 'labels': val_labels}, columns=col_names)
    return {'trn': df_trn, 'val': df_val}


def save_data(dataset: dict, folder: Path, supervised: bool, test_size: float=0.1):
    df_trn, df_val = [dataset[key].copy() for key in ('trn', 'val')]

    if not supervised:
        df_trn = df_trn.loc[df_trn['labels'] != 2]
        trn_texts, val_texts = [df['text'].values for df in (df_trn, df_val)]
        concat_texts = np.concatenate([trn_texts, val_texts])
        trn_texts, val_texts = train_test_split(concat_texts, test_size=test_size)
        df_trn = pd.DataFrame({'text': trn_texts, 'labels': [0]*len(trn_texts)}, columns=df_trn.columns)

    df_trn.to_csv(folder/'train.csv', header=False, index=False)
    df_val.to_csv(folder/'test.csv', header=False, index=False)
    (folder/'classes.txt').open('w').writelines(f'{c}\n' for c in CLASSES)


def load_data(path, chunksize=None):
    df_trn = pd.read_csv(path/'train.csv', header=None, chunksize=chunksize)
    df_val = pd.read_csv(path/'test.csv', header=None, chunksize=chunksize)
    return {'trn': df_trn, 'val': df_val}


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return RE_SPACE.sub(' ', html.unescape(x))


if __name__ == '__main__':
    main()
