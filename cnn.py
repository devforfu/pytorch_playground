import cv2 as cv
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from core.utils import LabelledImagesDataset


def imread(filename):
    img = cv.imread(str(filename))
    converted = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return converted.transpose(2, 0, 1)


def as_tensor(x, y):
    return torch.FloatTensor(x).cuda(), torch.LongTensor(y).cuda()


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


class IdentityBlock(nn.Module):

    def __init__(self, ni, nf, kernel):
        super().__init__()
        self.bn_in = nn.BatchNorm2d(ni)
        self.conv1 = nn.Conv2d(ni, nf, kernel, stride=1)
        self.bn1 = nn.BatchNorm2d(ni)
        self.conv2 = nn.Conv2d(ni, nf, kernel, stride=1)
        self.bn2 = nn.BatchNorm2d(ni)

    def forward(self, x):
        shortcut = self.bn_in(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += shortcut
        x = F.leaky_relu(x)
        return x


class UpsamplingBlock(nn.Module):

    def __init__(self, ni, nf, kernel, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(ni, nf, 1, stride=stride)
        self.bn1 = nn.BatchNorm2d(ni)
        self.conv2 = nn.Conv2d(ni, nf, kernel, stride=stride)
        self.bn2 = nn.BatchNorm2d(ni)
        self.conv3 = nn.Conv2d(ni, nf, kernel, stride=stride)
        self.bn3 = nn.BatchNorm2d(ni)

    def forward(self, x):
        shortcut = self.bn1(self.conv1(x))

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += shortcut
        x = F.leaky_relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, ni, kernel=(7, 7)):
        super().__init__()
        self.conv = nn.Conv2d(ni, 64, kernel, stride=2)
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)

        self.block1 = nn.Sequential(
            IdentityBlock(ni, 64, 3),
            IdentityBlock(64, 64, 3),
            IdentityBlock(64, 64, 3),
        )

        self.block2 = nn.Sequential(
            UpsamplingBlock(64, 128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.block1(x)
        return x


def main():
    path = '/home/ck/data/cifar10/train'
    dataset = LabelledImagesDataset(
        labels_from='folders', root=path, batch_size=2048,
        one_hot=False, reader=imread)

    # model = ConvNet([3, 20, 40, 80], 10).cuda()
    model = ResNet(3)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, 101):
        epoch_loss = 0
        steps = 0
        for batch in dataset:
            x, y = as_tensor(*batch)
            steps += 1
            out = model(x)
            optimizer.zero_grad()
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= steps
        print('Epoch %03d train loss: %2.4f' % (epoch, epoch_loss))


if __name__ == '__main__':
    main()
