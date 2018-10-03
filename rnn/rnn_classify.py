import re
import html
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fastai.text import Tokenizer, partition_by_cores


BOS = 'xbox'   # beginning-of-sentence tag
FLD = 'xfld'  # field-data tag

DATA_ROOT = Path.home() / 'data'
IMDB = DATA_ROOT / 'aclImdb'
SENTINEL = IMDB / 'created'
CLASS_PATH = IMDB / 'imdb_class'
LM_PATH =  IMDB / 'imdb_lm'
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
        SENTINEL.open('w').write('1')

    cls_data, lm_data = [load_data(path) for path in (CLASS_PATH, LM_PATH)]

    tokens_path = LM_PATH / 'tmp' / 'tokens.pickle'

    if tokens_path.exists():
        with tokens_path.open('rb') as file:
            data = pickle.load(file)

    else:
        df_trn, df_val = lm_data['trn'], lm_data['val']
        tokens_trn, labels_trn = tokenize(df_trn)
        tokens_val, labels_val = tokenize(df_val)
        with tokens_path.open('wb') as file:
            data = {
                'tokens_trn': tokens_trn,
                'labels_trn': labels_val,
                'tokens_val': tokens_val,
                'labels_val': labels_val}
            pickle.dump(data, file)

    freq = Counter(token for sentence in data['tokens_trn'] for token in sentence)
    print(freq.most_common(25))


def create_if_not_exist(*folders):
    for path in folders:
        if not path.exists():
            path.mkdir(parents=True)


def create_dataset(root):
    trn_texts, trn_labels = read_texts(root / 'train')
    val_texts, val_labels = read_texts(root / 'test')

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


def read_texts(path):
    texts, labels = [], []
    for index, label in enumerate(CLASSES):
        for filename in (path/label).glob('*.*'):
            texts.append(filename.open('r').read())
            labels.append(index)
    return np.array(texts), np.array(labels)


def save_data(dataset: dict, folder: Path, supervised: bool, test_size: float=0.1):
    df_trn, df_val = [dataset[key].copy() for key in ('trn', 'val')]

    if supervised:
        df_trn = df_trn.loc[df_trn['labels'] != 2]
        (folder / 'classes.txt').open('w').writelines(f'{c}\n' for c in CLASSES)
    else:
        trn_texts, val_texts = [df['text'].values for df in (df_trn, df_val)]
        concat_texts = np.concatenate([trn_texts, val_texts])
        trn_texts, val_texts = train_test_split(concat_texts, test_size=test_size, random_state=RANDOM_SEED)
        df_trn = pd.DataFrame({'text': trn_texts, 'labels': [0] * len(trn_texts)}, columns=df_trn.columns)
        df_val = pd.DataFrame({'text': val_texts, 'labels': [0] * len(val_texts)}, columns=df_val.columns)

    df_trn.to_csv(folder/'train.csv', header=False, index=False)
    df_val.to_csv(folder/'test.csv', header=False, index=False)


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


def tokenize(data, n_labels=1):

    def read_chunk(df):
        labels = df.iloc[:, range(n_labels)].values.astype(np.int64)
        texts = f'\n{BOS} {FLD} 1 {df[n_labels].astype(str)}'
        for i in range(n_labels + 1, len(df.columns)):
            texts += f' {FLD} {i - n_labels} {df[i].astype(str)}'
        texts = list(texts.apply(fixup).values)
        tokenizer = Tokenizer()
        tokens = tokenizer.proc_all_mp(partition_by_cores(texts))
        return tokens, list(labels)

    all_tokens, all_labels = [], []
    for index, row in enumerate(data):
        chunk_tokens, chunk_labels = read_chunk(row)
        all_tokens += chunk_tokens
        all_labels += chunk_labels

    return all_tokens, all_labels


if __name__ == '__main__':
    main()
