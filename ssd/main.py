import json
from pathlib import Path
from itertools import chain, islice
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from torchvision import transforms

from utils import open_image


ROOT = Path.home().joinpath('data', 'voc2007')
TRAIN_JSON = ROOT / 'pascal_train2007.json'
TRAIN_JPEG = ROOT.joinpath('VOCdevkit', 'VOC2007', 'JPEGImages')


class VOCDataset(Dataset):

    def __init__(self, json_path, images_path, size=224, augmentations=None):
        self.json_path = json_path
        self.images_path = images_path
        self.size = size
        self.transform = build_transform(augmentations)
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
        id2cat = {i: c for i, c in enumerate(categories, 1)}
        cat2id = {c: i for i, c in enumerate(categories, 1)}

        samples = []
        for file_name, group in dataset.groupby('file_name'):
            boxes = list(chain.from_iterable(group.bbox))
            classes = [cat2id[name] for name in group.category_name]
            samples.append((file_name, boxes, classes))
        df = pd.DataFrame(samples, columns=['file_name', 'boxes', 'classes'])

        self.id2cat = id2cat
        self.cat2id = cat2id
        self._dataset = df

    def __getitem__(self, index):
        """
        Note that index could be a single integer, or a batch of indexes.
        """
        records = list(self._dataset.loc[index].itertuples())
        np_images = [self.open(r.file_name) for r in records]
        images = [self.transform(image) for image in np_images]
        boxes, classes = list(zip(*[(r.boxes, r.classes) for r in records]))
        boxes, classes = pad(boxes), pad(classes)
        return torch.stack(images), t(boxes), t(classes)

    def __len__(self):
        return len(self._dataset)

    def open(self, file_name):
        return open_image(self.images_path / file_name, self.size)


def t(obj):
    return torch.tensor(obj)


def build_transform(augmentations=None):
    transforms_list = augmentations or []
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)


class VOCDataLoader:

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False,
                 num_workers=0, transforms=None):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.batch_sampler = self._get_sampler()

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        iterator = iter(self.batch_sampler)
        if self.num_workers > 0:
            n = self.num_workers * 10
            chunks = islice(iterator, 0, n)
            with ThreadPoolExecutor(self.num_workers) as executor:
                yield from executor.map(self._get_batch, chunks)
        else:
            for indexes in iterator:
                yield self._get_batch(indexes)

    def _get_sampler(self):
        sampler_class = RandomSampler if self.shuffle else SequentialSampler
        sampler = sampler_class(self.dataset)
        batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last)
        return batch_sampler

    def _get_batch(self, indexes):
        return self.dataset[indexes]


def pad(arr, pad_value=0):
    longest = len(max(arr, key=len))
    return [
        [pad_value for _ in range(longest - len(vec))] + vec
        for vec in arr]


def main():
    dataset = VOCDataset(TRAIN_JSON, TRAIN_JPEG)
    loader = VOCDataLoader(dataset, batch_size=4, num_workers=cpu_count())
    for batch in iter(loader):
        print(batch[2])



if __name__ == '__main__':
    main()
