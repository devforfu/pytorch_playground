from dataclasses import dataclass
from pathlib import Path
import pickle
from random import shuffle
import shutil

from fastai.text import TextDataset
from torch.utils.data import Dataset


DATA_ROOT = Path.home() / 'data'
LYRICS_PATH = DATA_ROOT / 'azlyrics' / 'many'


def main():
    meta = prepare_lyrics(LYRICS_PATH, LYRICS_PATH.parent/'prepared')
    dataset = TextDataset.from_folder(meta.folder)
    print(f'Dataset size: {len(dataset)}')


def prepare_lyrics(src, dst, test_size: float=0.2) -> 'LyricsData':
    meta = dst/'meta.pickle'

    if meta.exists():
        with meta.open('rb') as file:
            return pickle.load(file)

    classes, songs = [], {}

    for subdir in src.glob('*'):
        author = subdir.stem
        classes.append(author)
        author_songs = {}
        with (subdir/'songs.csv').open() as file:
            for line in file:
                index, _, name = line.partition(',')
                author_songs[int(index)] = name.strip()
        songs[author] = author_songs

        files = list(subdir.glob('*.txt'))
        shuffle(files)
        sz = int(len(files) * (1 - test_size))
        train, test = files[:sz], files[sz:]

        train_dir = dst/'train'/author
        train_dir.mkdir(parents=True, exist_ok=True)
        for txt_file in train:
            shutil.copy(txt_file, train_dir/txt_file.name)

        test_dir = dst/'test'/author
        test_dir.mkdir(parents=True, exist_ok=True)
        for txt_file in test:
            shutil.copy(txt_file, test_dir/txt_file.name)

    data = LyricsData(dst, classes, songs)

    with meta.open('wb') as file:
        pickle.dump(data, file)

    return data


@dataclass
class LyricsData:
    folder: str
    classes: list
    songs: dict


if __name__ == '__main__':
    main()
