import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms

from core.loop import Loop
from core.callbacks import Logger
from core.schedule import CosineAnnealingLR
from core.utils import LabelledImagesDataset


class ConvNet(nn.Module):

    def __init__(self, layers, outputs):
        super().__init__()
        n = len(layers) - 1
        self.layers = nn.ModuleList([
            nn.Conv2d(layers[i], layers[i + 1], kernel_size=3, stride=2)
            for i in range(n)
        ])
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(layers[-1], outputs)

    def forward(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        x = self.pool(x)
        x = torch.squeeze(x)
        x = self.out(x)
        return F.log_softmax(x, dim=-1)


def conv3x3(ni, nf, stride=1):
    return nn.Conv2d(ni, nf, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
                nn.BatchNorm2d(nf)
            )

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'downsample'):
            shortcut = self.downsample(shortcut)

        out += shortcut
        out = F.leaky_relu(out)
        return out


class Downsample(nn.Module):

    def __init__(self, ni, nf, stride):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=1, stride=stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(nf)

    def forward(self, x):
        return self.bn(self.conv(x))


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
        return F.log_softmax(x, dim=-1)


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


class SimpleResNet(nn.Module):

    def __init__(self, layers, num_of_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers1 = nn.ModuleList([
            ConvLayer(x, y) for (x, y) in pairs(layers)
        ])
        self.layers2 = nn.ModuleList([
            ResNetLayer(x, x, 1) for x in layers[1:]
        ])
        self.layers3 = nn.ModuleList([
            ResNetLayer(x, x, 1) for x in layers[1:]
        ])
        self.fc = nn.Linear(layers[-1], num_of_classes)

    def forward(self, x):
        x = self.conv(x)
        for l1, l2, l3 in zip(self.layers1, self.layers2, self.layers3):
            x = l3(l2(l1(x)))
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


def pairs(xs):
    current, *rest = xs
    for item in rest:
        yield current, item
        current = item


def imread(filename):
    img = cv.imread(str(filename))
    converted = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return converted.transpose(2, 0, 1)


def as_tensor(x, y):
    return torch.FloatTensor(x).cuda(), torch.LongTensor(y).cuda()


def to_xy(x, y):
    images = np.stack([imread(filename) for filename in x])
    classes = np.argmax(y, axis=1)
    return images, classes


def main():
    path = '/home/ck/data/cifar10/train'

    dataset = LabelledImagesDataset(
        labels_from='folders', root=path, batch_size=2048, one_hot=False,
        transforms=[to_xy, as_tensor])

    train_data = iter(dataset)
    n = len(train_data)

    model = SimpleResNet([10, 20, 40, 80, 160], 10).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    schedule = CosineAnnealingLR(optimizer, t_max=n/2, eta_min=1e-5)
    loop = Loop(model, optimizer, schedule)

    loop.run(train_data=train_data, callbacks=[Logger()])


if __name__ == '__main__':
    main()
