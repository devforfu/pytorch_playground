import math
from pathlib import Path
from multiprocessing import cpu_count

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST
from torchvision.models.resnet import resnet18
from torchvision.utils import make_grid
import onnx
from onnx import onnx_pb
import onnx_coreml
from onnx_coreml import convert

from core.loop import Loop
from core.metrics import accuracy
from core.callbacks import default_callbacks


DATA_ROOT = Path.home() / 'data' / 'emnist'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
STATS = [0.17325], [0.33163]


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


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNet(nn.Module):

    def __init__(self, num_of_classes):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=2)
        self.blocks = nn.ModuleList([
            IdentityBlock(10, 20, stride=2),
            IdentityBlock(20, 40, stride=2),
            IdentityBlock(40, 80, stride=2)
        ])
        self.pool = nn.AvgPool2d(4)
        self.flatten = Flatten()
        self.fc = nn.Linear(80, num_of_classes)
        self.init()

    def forward(self, x):
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def load_dataset(data_transforms, root=DATA_ROOT, split='digits',
                 batch_size=1024, num_workers=0):

    datasets = {}
    for name in ('train', 'valid'):
        is_training = name == 'train'
        dataset = EMNIST(
            root=root, split=split, train=is_training, download=True,
            transform=data_transforms[name])
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers)
        datasets[name] = {'dataset': dataset, 'loader': loader}
    return datasets


def random_sample(dataset, n=16):
    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    return next(iter(loader))


def compute_stats(dataset):
    n = len(dataset) // 1000
    loader = DataLoader(
        dataset,
        batch_size=n,
        num_workers=cpu_count())
    mean, std, total = 0., 0., 0
    for batch, _ in iter(loader):
        image = batch.squeeze()
        mean += image.mean().item()
        std += image.std().item()
        total += 1
    mean /= total
    std /= total
    print(mean, std)


def show_predictions(images, suptitle='', titles=None, dims=(4, 4), figsize=(12, 12)):
    f, ax = plt.subplots(*dims, figsize=figsize)
    titles = titles or []
    f.suptitle(suptitle)
    [mean], [std] = STATS
    images *= mean
    images += std
    for i, (img, ax) in enumerate(zip(images, ax.flat)):
        ax.imshow(img.reshape(28, 28))
        if i < len(titles):
            ax.set_title(titles[i])
    plt.show()


def to_np(*tensors):

    def convert_to_numpy(obj):
        return obj.detach().cpu().numpy()

    if len(tensors) == 1:
        return convert_to_numpy(tensors[0])
    return [convert_to_numpy(tensor) for tensor in tensors]


def main():
    batch_size = 10000
    num_workers = cpu_count()
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(4),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(*STATS)
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*STATS)
        ])
    }
    datasets = load_dataset(
        data_transforms,
        batch_size=batch_size,
        num_workers=num_workers)

    n_samples = len(datasets['train']['loader'])
    n_batches = math.ceil(n_samples / batch_size)

    model = ResNet(10)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    sched = CosineAnnealingLR(opt, T_max=n_batches/4, eta_min=1e-5)
    loop = Loop(model, opt, sched, device=DEVICE)

    loop.run(train_data=datasets['train']['loader'],
             valid_data=datasets['valid']['loader'],
             loss_fn=F.cross_entropy,
             metrics=[accuracy],
             callbacks=default_callbacks(),
             epochs=3)

    best_model = loop['Checkpoint'].best_model
    weights = torch.load(best_model)
    model.load_state_dict(weights)
    x, y = random_sample(datasets['valid']['dataset'])
    y_pred = model(x.to(DEVICE))
    valid_acc = accuracy(y_pred, y.to(DEVICE))
    title = f'Validation accuracy: {valid_acc:2.2%}'
    titles = [str(x) for x in to_np(y_pred.argmax(dim=1))]

    show_predictions(
        images=to_np(x.permute(0, 3, 2, 1)),
        suptitle=title,
        titles=titles)

    dummy_input = torch.randn(16, 1, 28, 28, requires_grad=True).cuda()
    torch.onnx.export(model, dummy_input, 'trivial.onnx', export_params=True)
    core_ml_model = convert('digits.onnx')
    core_ml_model.save('digits.mlmodel')
    print('CoreML model was saved onto disk')


if __name__ == '__main__':
    main()
