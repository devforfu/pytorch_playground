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
from core.schedule import CosineAnnealingLR
from core.callbacks import (
    Logger, History, EarlyStopping, CSVLogger, Checkpoint)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MEAN = np.array([0.4914, 0.48216, 0.44653])
STD = np.array([0.24703, 0.24349, 0.26159])


def conv3x3(ni, nf, stride=1, padding=1):
    return nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=padding,
                     bias=False)


class IdentityBlock(nn.Module):

    def __init__(self, ni, nf=None, stride=1):
        super().__init__()

        nf = ni if nf is None else nf
        self.conv1 = conv3x3(ni, nf, stride=stride)
        self.bn1 = nn.BatchNorm2d(nf)
        self.conv2 = conv3x3(nf, nf)
        self.bn2 = nn.BatchNorm2d(nf)

        if ni != nf:
            self.downsample = nn.Sequential(
                nn.Conv2d(ni, nf, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(nf))

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'downsample'):
            shortcut = self.downsample(x)

        out += shortcut
        out = F.leaky_relu(out)

        return out


class CustomResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.block1 = IdentityBlock(10, 20, stride=2)
        self.block2 = IdentityBlock(20, 40, stride=2)
        self.block3 = IdentityBlock(40, 80, stride=2)
        self.block4 = IdentityBlock(80, 160, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(160, 10)
        self.init()

    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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


def accuracy(y_pred, y_true):
    match = y_pred.argmax(dim=1) == y_true
    acc = match.type(torch.float).mean()
    return acc.item()


def main():
    root = Path.home() / 'data' / 'cifar10'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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
        training = name == 'train'
        datasets[name] = dataset
        loaders[name] = DataLoader(
            dataset=dataset, batch_size=256,
            shuffle=training, num_workers=cpu_count())
        dataset_sizes[name] = len(dataset)

    n = len(datasets['train'])

    model = CustomResNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    schedule = CosineAnnealingLR(optimizer, t_max=n, eta_min=1e-5, cycle_mult=2)
    loop = Loop(model, optimizer, schedule, device=DEVICE)

    callbacks = [
        History(), CSVLogger(), Logger(),
        EarlyStopping(patience=50), Checkpoint()]

    loop.run(
        train_data=loaders['train'],
        valid_data=loaders['valid'],
        callbacks=callbacks,
        loss_fn=F.cross_entropy,
        metrics=[accuracy],
        epochs=150)

    dataset = datasets['valid']
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    x, y = next(iter(loader))
    state = torch.load(loop['Checkpoint'].best_model)
    model.load_state_dict(state)
    predictions = model(x.cuda())
    labels = predictions.argmax(dim=1)
    verbose = [dataset.classes[name] for name in labels]
    imshow(utils.make_grid(x), title=verbose)


if __name__ == '__main__':
    main()
