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
from core.utils import LabelledImagesDataset
from core.callbacks import (
    Logger, History, EarlyStopping, CSVLogger, Checkpoint)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MEAN = np.array([0.4914, 0.48216, 0.44653])
STD = np.array([0.24703, 0.24349, 0.26159])


# class ConvLayer(nn.Module):
#
#     def __init__(self, ni, nf, stride=2, kernel_size=3):
#         super().__init__()
#         self.conv = nn.Conv2d(
#             in_channels=ni, out_channels=nf,
#             kernel_size=kernel_size, stride=stride,
#             bias=False, padding=1)
#         self.bn = nn.BatchNorm2d(nf)
#
#     def forward(self, x):
#         return F.leaky_relu(self.bn(self.conv(x)))
#
#
# class ResNetLayer(ConvLayer):
#
#     def forward(self, x):
#         return x + super().forward(x)
#
#
# class FastAIResNet(nn.Module):
#
#     def __init__(self, layers, num_of_classes):
#         super().__init__()
#         self.conv = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
#         self.layers1 = nn.ModuleList([
#             ConvLayer(x, y) for (x, y) in pairs(layers)])
#         self.layers2 = nn.ModuleList([
#             ResNetLayer(x, x, 1) for x in layers[1:]])
#         self.layers3 = nn.ModuleList([
#             ResNetLayer(x, x, 1) for x in layers[1:]])
#         self.fc = nn.Linear(layers[-1], num_of_classes)
#
#     def forward(self, x):
#         x = self.conv(x)
#         for l1, l2, l3 in zip(self.layers1, self.layers2, self.layers3):
#             x = l3(l2(l1(x)))
#         x = F.adaptive_max_pool2d(x, 1)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=-1)


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


class ConvNet(nn.Module):

    def __init__(self, layers, c):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([
            nn.Conv2d(layers[i], layers[i + 1], kernel_size=3, stride=2)
            for i in range(len(layers) - 1)])
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(layers[-1], c)

    def forward(self, x):
        x = self.conv(x)
        for l in self.layers:
            x = F.relu(l(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


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


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)

        self.block1 = nn.Sequential(
            IdentityBlock(16),
            IdentityBlock(16),
            IdentityBlock(16)
        )

        self.block2 = nn.Sequential(
            IdentityBlock(16, 32, stride=2),
            IdentityBlock(32),
            IdentityBlock(32),
            IdentityBlock(32)
        )

        self.block3 = nn.Sequential(
            IdentityBlock(32, 64, stride=2),
            IdentityBlock(64),
            IdentityBlock(64),
            IdentityBlock(64),
            IdentityBlock(64)
        )

        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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


class ConvLayer(nn.Module):

    def __init__(self, ni, nf, stride=2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            ni, nf, kernel_size=kernel_size,
            stride=stride, bias=False, padding=1)
        self.bn = nn.BatchNorm2d(nf)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNetLayer(ConvLayer):

    def forward(self, x):
        return x + super().forward(x)


class FastAIResNet(nn.Module):

    def __init__(self, layers, num_of_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ConvLayer(x, y),
                ResNetLayer(y, y, stride=1),
                ResNetLayer(y, y, stride=1))
            for x, y in pairs(layers)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(layers[-1], num_of_classes)

    def forward(self, x):
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# class BnLayer(nn.Module):
#
#     def __init__(self, ni, nf, stride=2, kernel_size=3):
#         super().__init__()
#         self.conv = nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride,
#                               bias=False, padding=1)
#         self.a = nn.Parameter(torch.zeros(nf, 1, 1))
#         self.m = nn.Parameter(torch.ones(nf, 1, 1))
#
#     def forward(self, x):
#         x = F.relu(self.conv(x))
#         x_chan = x.transpose(0, 1).contiguous().view(x.size(1), -1)
#         if self.training:
#             self.means = x_chan.mean(1)[:, None, None]
#             self.stds = x_chan.std(1)[:, None, None]
#         return (x - self.means) / self.stds * self.m + self.a


class BnLayer(nn.Module):

    def __init__(self, ni, nf, stride=2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride,
                              bias=False, padding=1)
        self.bn = nn.BatchNorm2d(nf)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResnetLayer(BnLayer):

    def forward(self, x): return x + super().forward(x)


class Resnet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i + 1])
                                     for i in range(len(layers) - 1)])
        self.layers2 = nn.ModuleList([ResnetLayer(layers[i + 1], layers[i + 1], 1)
                                      for i in range(len(layers) - 1)])
        self.layers3 = nn.ModuleList([ResnetLayer(layers[i + 1], layers[i + 1], 1)
                                      for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)

    def forward(self, x):
        x = self.conv1(x)
        for l, l2, l3 in zip(self.layers, self.layers2, self.layers3):
            x = l3(l2(l(x)))
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.out(x)


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
            shuffle=training, num_workers=0)
        dataset_sizes[name] = len(dataset)

    # model = ResNet()
    # model = ConvNet([10, 20, 40, 80, 160], 10)
    # model = CustomResNet()
    # model = FastAIResNet([10, 20, 40, 80, 160], 10)

    model = FastAIResNet([10, 20, 40, 80, 160], 10)
    # model = CustomResNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    n = len(datasets['train'])
    # optimizer = optim.SGD(
    #     model.parameters(), momentum=0.9, weight_decay=1e-5, lr=1e-2)

    schedule = CosineAnnealingLR(optimizer, t_max=n, eta_min=1e-5, cycle_mult=2)
    loop = Loop(model, optimizer, schedule, device=DEVICE)
    callbacks = [
        History(), CSVLogger(), Logger(),
        EarlyStopping(patience=3), Checkpoint()]

    loop.run(
        train_data=loaders['train'],
        valid_data=loaders['valid'],
        callbacks=callbacks,
        loss_fn=F.cross_entropy,
        metrics=[accuracy],
        epochs=5)


if __name__ == '__main__':
    main()
