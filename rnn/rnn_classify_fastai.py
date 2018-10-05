import re
import html
import pickle
from pathlib import Path
from functools import partial
from collections import Counter, defaultdict, namedtuple

import numpy as np
import pandas as pd
from torch import optim
from sklearn.model_selection import train_test_split

from fastai.text import accuracy
from fastai.text import Tokenizer, partition_by_cores, get_rnn_classifier
from fastai.text import LanguageModelLoader, LanguageModelData
from fastai.text import TextDataset, SortishSampler, SortSampler, DataLoader, ModelData
from fastai.text import RNN_Learner, TextModel, to_gpu, seq2seq_reg


BOS = 'xbox'  # beginning-of-sentence tag
FLD = 'xfld'  # field-data tag

DATA_ROOT = Path.home() / 'data'
IMDB = DATA_ROOT / 'aclImdb'
SENTINEL = IMDB / 'created'

LM_PATH =  IMDB / 'imdb_lm'
LM_TOKENS_PATH = LM_PATH / 'tmp' / 'tokens.pickle'
LM_VOCAB_PATH = LM_PATH / 'tmp' / 'vocab.pickle'

CLASS_PATH = IMDB / 'imdb_class'
CLS_TOKENS_PATH = CLASS_PATH / 'tmp' / 'tokens.pickle'
CLS_VOCAB_PATH = CLASS_PATH / 'tmp' / 'vocab.pickle'
CLASSES = ['neg', 'pos', 'unsup']

RE_SPACE = re.compile(r'  +')
RANDOM_SEED = 42

Vocab = namedtuple('Vocab', 'itos stoi size')


def main():
    np.random.seed(RANDOM_SEED)

    if not SENTINEL.exists():
        create_if_not_exist(CLASS_PATH, LM_PATH)
        dataset = create_dataset(IMDB)
        save_data(dataset, CLASS_PATH, supervised=True)
        save_data(dataset, LM_PATH, supervised=False)
        SENTINEL.open('w').write('1')

    cls_data, lm_data = [load_data(path) for path in (CLASS_PATH, LM_PATH)]

    data = maybe_create_tokens(LM_TOKENS_PATH, lm_data)
    vocab, lm_trn, lm_val = maybe_create_vocab(LM_VOCAB_PATH, data)

    # lr = 1e-3
    # wd = 1e-7
    # bptt = 70
    # bs = 52
    # em_sz, nh, nl = 400, 1150, 3
    # opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    # lrs = lr
    #
    # dl_trn = LanguageModelLoader(np.concatenate(lm_trn), bs, bptt)
    # dl_val = LanguageModelLoader(np.concatenate(lm_val), bs, bptt)
    # md = LanguageModelData(IMDB, 1, vocab.size, dl_trn, dl_val, bs=bs, bptt=bptt)
    # drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7
    # dropouts = dict(
    #     dropouti=drops[0],
    #     dropout=drops[1],
    #     wdrop=drops[2],
    #     dropoute=drops[3],
    #     dropouth=drops[4])
    #
    # learner = md.get_model(opt_fn, em_sz, nh, nl, **dropouts)
    # learner.metrics = [accuracy]
    # learner.fit(lrs/2, 1, wds=wd, use_clr=(32, 2), cycle_len=1)
    # learner.fit(lrs, 1, wds=wd, use_clr=(20, 10), cycle_len=15)
    # learner.save('lm1')
    # learner.save_encoder('lm1_enc')

    data = maybe_create_tokens(CLS_TOKENS_PATH, cls_data)
    vocab, cls_trn, cls_val = maybe_create_vocab(CLS_VOCAB_PATH, data)

    labels_trn = np.squeeze(data['labels_trn'])
    labels_val = np.squeeze(data['labels_val'])
    min_label = labels_trn.min()
    labels_trn -= min_label
    labels_val -= min_label
    c = int(labels_trn.max()) + 1

    bptt, em_sz, nh, nl = 70, 400, 1150, 3
    dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])
    bs = 48

    ds_trn = TextDataset(cls_trn, labels_trn)
    ds_val = TextDataset(cls_val, labels_val)
    sampler_trn = SortishSampler(cls_trn, key=lambda x: len(cls_trn[x]), bs=bs//2)
    sampler_val = SortSampler(cls_val, key=lambda x: len(cls_val[x]))
    dl_trn = DataLoader(ds_trn, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=sampler_trn)
    dl_val = DataLoader(ds_val, bs, transpose=True, num_workers=1, pad_idx=1, sampler=sampler_val)
    md = ModelData(IMDB, dl_trn, dl_val)

    m = get_rnn_classifier(
        bptt, 20 * 70, c, vocab.size,
        emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
        layers=[em_sz * 3, 50, c], drops=[dps[4], 0.1],
        dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
    learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip = 25.
    learn.metrics = [accuracy]

    lr = 3e-3
    lrm = 2.6
    lrs = np.array([lr / (lrm ** 4), lr / (lrm ** 3), lr / (lrm ** 2), lr / lrm, lr])

    wd = 1e-7
    # learn.load_encoder('lm1_enc')
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))


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


def load_data(path, chunksize=24000):
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
        texts = f'\n{BOS} {FLD} 1 ' + df[n_labels].astype(str)
        for i in range(n_labels + 1, len(df.columns)):
            texts += f' {FLD} {i - n_labels} ' + df[i].astype(str)
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


def create_vocab(data, max_size: int=60000, min_freq: int=2) -> Vocab:
    counts = Counter(token for sentence in data for token in sentence)
    itos = [
        char for char, count in counts.most_common(max_size)
        if count > min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    stoi = defaultdict(lambda: 0, {v:k for k,  v in enumerate(itos)})
    return Vocab(itos=itos, stoi=stoi, size=len(itos))


def maybe_create_tokens(tokens_path, lm_data):
    if tokens_path.exists():
        with tokens_path.open('rb') as file:
            return pickle.load(file)

    tokens_path.parent.mkdir(exist_ok=True)
    df_trn, df_val = lm_data['trn'], lm_data['val']
    tokens_trn, labels_trn = tokenize(df_trn)
    tokens_val, labels_val = tokenize(df_val)

    with tokens_path.open('wb') as file:
        data = {
            'tokens_trn': tokens_trn,
            'labels_trn': labels_trn,
            'tokens_val': tokens_val,
            'labels_val': labels_val}
        pickle.dump(data, file)

    return data


def maybe_create_vocab(vocab_path, data):
    if vocab_path.exists():
        with vocab_path.open('rb') as file:
            itos, stoi, lm_trn, lm_val = [pickle.load(file) for _ in range(4)]
        stoi = defaultdict(lambda: 0, stoi)
        vocab = Vocab(itos=itos, stoi=stoi, size=len(itos))

    else:
        vocab_path.parent.mkdir(exist_ok=True)
        vocab = create_vocab(data['tokens_trn'])
        lm_trn, lm_val = [
            np.array([
                [vocab.stoi[token] for token in sentence]
                for sentence in data[key]
            ]) for key in ('tokens_trn', 'tokens_val')]
        with vocab_path.open('wb') as file:
            for obj in (vocab.itos, dict(vocab.stoi), lm_trn, lm_val):
                pickle.dump(obj, file)

    return vocab, lm_trn, lm_val


if __name__ == '__main__':
    main()
