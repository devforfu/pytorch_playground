from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

from core.loop import Loop
from core.callbacks import Logger
from core.schedule import CosineAnnealingLR
from core.utils import LabelledImagesDataset


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MEAN = np.array([0.4914, 0.48216, 0.44653])
STD = np.array([0.24703, 0.24349, 0.26159])


class ConvLayer(nn.Module):

    def __init__(self, ni, nf, stride=2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=ni, out_channels=nf,
            kernel_size=kernel_size, stride=stride,
            bias=False, padding=1)
        self.bn = nn.BatchNorm2d(nf)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)))


class ResNetLayer(ConvLayer):

    def forward(self, x):
        return x + super().forward(x)


class FastAIResNet(nn.Module):

    def __init__(self, layers, num_of_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers1 = nn.ModuleList([
            ConvLayer(x, y) for (x, y) in pairs(layers)])
        self.layers2 = nn.ModuleList([
            ResNetLayer(x, x, 1) for x in layers[1:]])
        self.layers3 = nn.ModuleList([
            ResNetLayer(x, x, 1) for x in layers[1:]])
        self.fc = nn.Linear(layers[-1], num_of_classes)

    def forward(self, x):
        x = self.conv(x)
        for l1, l2, l3 in zip(self.layers1, self.layers2, self.layers3):
            x = l3(l2(l1(x)))
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        # return F.log_softmax(x, dim=-1)


def pairs(xs):
    current, *rest = xs
    for item in rest:
        yield current, item
        current = item


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
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.Pad(4),
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

    # class_names = datasets['train'].classes
    # samples, targets = next(iter(loaders['train']))
    # out = utils.make_grid(samples)
    # imshow(out, title=[class_names[x] for x in targets])

    model = FastAIResNet([10, 20, 40, 80, 160], 10)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    schedule = CosineAnnealingLR(
        optimizer, t_max=len(datasets['train']), eta_min=1e-5)

    loop = Loop(model, optimizer, schedule)
    loop.run(train_data=iter(loaders['train']),
             valid_data=iter(loaders['valid']),
             loss_fn=F.cross_entropy,
             callbacks=[Logger()])


if __name__ == '__main__':
    main()
