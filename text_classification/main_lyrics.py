from dataclasses import dataclass
from pathlib import Path
import shutil

from fastai.text import TextDataset
from torch.utils.data import Dataset


DATA_ROOT = Path.home() / 'data'
LYRICS_PATH = DATA_ROOT / 'azlyrics' / 'many'


def main():
    prepare_lyrics(LYRICS_PATH, LYRICS_PATH.parent/'prepared')
    dataset = TextDataset.from_folder(LYRICS_PATH)
    print(f'Dataset size: {len(dataset)}')


def prepare_lyrics(src, dst, test_size: float=0.2):
    classes, songs = [], {}

    for subdir in src.glob('*'):
        author = subdir.stem
        classes.append(author)
        author_songs = {}
        with (subdir/'songs.csv').open() as file:
            for line in file:
                index, name = line.split(',')
                author_songs[int(index)] = name.strip()
        songs[author] = author_songs

        files = list(subdir.glob('*.txt'))
        sz = int(len(files) * (1 - test_size))
        train, test = files[:sz], files[sz:]

        train_dir = dst/'train'/author
        train_dir.mkdir(parents=True, exist_ok=True)
        for txt_file in train:
            shutil.copy(txt_file, train_dir/txt_file.name)

        test_dir = dst/'test'/author
        test_dir.mkdir(parents=True, exist_ok=True)
        for filename in test:
            shutil.copy(filename, test_dir/filename)

    return LyricsData(dst, classes, songs)


@dataclass
class LyricsData:
    folder: str
    classes: list
    songs: dict


if __name__ == '__main__':
    main()
