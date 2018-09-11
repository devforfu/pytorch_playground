from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MEAN = np.array([0.4914, 0.48216, 0.44653])
STD = np.array([0.24703, 0.24349, 0.26159])


def imshow(image, title=None):
    img = image.numpy().transpose((1, 2, 0))
    img = STD*img + MEAN
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def main():
    root = Path.home() / 'data' / 'cifar10'

    data_transforms = {
        'train': transforms.Compose([
            transforms.Pad(4),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    }

    datasets, loaders, dataset_sizes = {}, {}, {}
    for name in ('train', 'valid'):
        dataset = ImageFolder(root/name, data_transforms[name])
        shuffle = name == 'train'
        datasets[name] = dataset
        loaders[name] = DataLoader(
            dataset=dataset, batch_size=4,
            shuffle=shuffle, num_workers=cpu_count())
        dataset_sizes[name] = len(dataset)

    class_names = datasets['train'].classes
    samples, targets = next(iter(loaders['train']))
    out = utils.make_grid(samples)
    imshow(out, title=[class_names[x] for x in targets])


if __name__ == '__main__':
    main()
