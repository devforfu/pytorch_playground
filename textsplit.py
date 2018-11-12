"""
Splits folders with songs into training and validation subsets.
"""
import argparse
from pathlib import Path
import random

import pandas as pd


def main():
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    meta = []

    for subfolder in args.input.iterdir():
        print(f'Splitting folder for artist \'{subfolder.stem}\'')
        files = [file for file in subfolder.iterdir() if file.suffix == '.txt']
        n_train = int(len(files) * args.train_size)
        random.shuffle(files)
        train_files, valid_files = files[:n_train], files[n_train:]
        print(f'Training: {len(train_files)}, validation: {len(valid_files)}')
        train_ids = {file.stem for file in train_files}

        split = [
            ('train', train_files),
            ('valid', valid_files)]

        for line in (subfolder/'songs.csv').open():
            index, _, song = line.partition(',')
            meta.append({
                'id': int(index),
                'artist': subfolder.stem,
                'song': song.strip(),
                'valid': index not in train_ids
            })

        for sub, files in split:
            new_dir = args.output/sub/subfolder.stem
            new_dir.mkdir(parents=True, exist_ok=True)
            for old_file in files:
                new_file = new_dir/old_file.name
                new_file.open('w').write(old_file.open().read())

    pd.DataFrame(meta).to_json(
        args.output/'songs.json', orient='records', index=False)

    print('Files copied into folder ', args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='path to folders with labelled texts'
    )
    parser.add_argument(
        '-o', '--output',
        default=Path.home(),
        help='path to save separated files'
    )
    parser.add_argument(
        '-ts', '--train-size',
        default=0.8, type=float,
        help='amount of texts (per category) to keep for training'
    )
    parser.add_argument(
        '-rs', '--random-state',
        default=1, type=int,
        help='random state to use when taking training subset'
    )

    args = parser.parse_args()
    args.input = Path(args.input)
    args.output = Path(args.output)
    random.seed(args.random_state)

    return args


if __name__ == '__main__':
    main()
