import json
from pathlib import Path
from itertools import chain
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler

from utils import open_image


ROOT = Path.home().joinpath('data', 'voc2007')
TRAIN_JSON = ROOT / 'pascal_train2007.json'
TRAIN_JPEG = ROOT.joinpath('VOCdevkit', 'VOC2007', 'JPEGImages')


class VOCDataset(Dataset):

    def __init__(self, json_path, images_path):
        self.json_path = json_path
        self.images_path = images_path
        self.id2cat = None
        self.cat2id = None
        self._dataset = None
        self.init()

    def init(self):
        with open(self.json_path) as file:
            content = json.load(file)

        annotations_df = pd.DataFrame(content['annotations']).set_index('id')
        images_df = pd.DataFrame(content['images']).set_index('id')
        categories_df = pd.DataFrame(content['categories']).set_index('id')
        columns = ['area', 'segmentation', 'height', 'width', 'supercategory']

        df = pd.DataFrame(
            annotations_df.
            join(images_df, on='image_id').
            join(categories_df, on='category_id').
            drop(columns=columns)
        ).rename(columns={'name': 'category_name'})

        dataset = df.loc[df.ignore != 1].reset_index(drop=True)
        categories = df.category_name.unique()
        id2cat = {i: c for i, c in enumerate(categories)}
        cat2id = {c: i for i, c in enumerate(categories)}

        boxes, classes = [], []
        for name, group in dataset.groupby('file_name'):
            boxes.append(list(chain.from_iterable(group.bbox)))
            classes.append([cat2id[name] for name in group.category_name])

        longest_box = len(max(boxes, key=len))
        boxes_padded = [
            [0 for _ in range(longest_box - len(vec))] + vec
            for vec in boxes]

        longest_target = len(max(classes, key=len))
        classes_padded = [
            [0 for _ in range(longest_target - len(vec))] + vec
            for vec in classes]

        self.id2cat = id2cat
        self.cat2id = cat2id
        self._dataset = dataset

    def __getitem__(self, index):
        record = self._dataset.loc[index]
        image = open_image(self.images_path / record.file_name)
        box = np.array(record.bbox)
        category_index = self.cat2id[record.category_name]
        return image, box, category_index

    def __len__(self):
        return len(self._dataset)


class VOCDataLoader:

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False,
                 num_workers=0):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.batch_sampler = self._get_sampler()

    def _get_sampler(self):
        sampler_class = RandomSampler if self.shuffle else SequentialSampler
        sampler = sampler_class(self.dataset)
        batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last)
        return batch_sampler

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):

        def get_batch(indexes):
            arr = self.dataset[indexes]
            return torch.tensor(arr)

        iterator = iter(self.batch_sampler)
        if self.num_workers > 0:
            with ThreadPoolExecutor(self.num_workers) as executor:
                for indexes in iterator:
                    yield from executor.map(get_batch, indexes)
        else:
            for indexes in iterator:
                batch = get_batch(indexes)
                yield batch


def main():
    dataset = VOCDataset(TRAIN_JSON, TRAIN_JPEG)
    loader = VOCDataLoader(dataset, batch_size=4)
    for batch in iter(loader):
        print(batch)

    # with open(TRAIN_JSON) as file:
    #     train_json = json.load(file)
    # print(train_json)


if __name__ == '__main__':
    main()
