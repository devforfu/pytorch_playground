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
from onnx import onnx_pb
from onnx_coreml import convert

from core.loop import Loop
from core.metrics import accuracy
from core.callbacks import default_callbacks


DATA_ROOT = Path.home() / 'data' / 'emnist'


def load_dataset(data_transforms, root=DATA_ROOT, split='digits',
                 batch_size=1024):

    datasets = {}
    for name in ('train', 'valid'):
        is_training = name == 'train'
        dataset = EMNIST(
            root=root, split=split, train=is_training, download=True,
            transform=data_transforms[name])
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=cpu_count())
        datasets[name] = {'dataset': dataset, 'loader': loader}
    return datasets


def random_sample(dataset, n=16):
    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    return next(loader)


def show(images, subtitle='', titles=None, dims=(4, 4), figsize=(12, 12)):
    f, ax = plt.subplots(*dims, figsize=figsize)
    titles = titles or []
    f.subtitle(subtitle)
    for i, (img, ax) in enumerate(zip(images, ax.flat)):
        ax.imshow(img)
        if i < len(titles):
            ax.title(titles[i])
    plt.show()


def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    datasets = load_dataset(data_transforms)

    model = resnet18(True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    sched = CosineAnnealingLR(opt, T_max=1e-2, eta_min=1e-5)
    loop = Loop(model, opt, sched)

    loop.run(train_data=datasets['train']['loader'],
             valid_data=datasets['valid']['loader'],
             loss_fn=F.cross_entropy,
             metrics=[accuracy],
             callbacks=default_callbacks(),
             epochs=10)

    best_model = loop['Checkpoint'].best_model
    weights = torch.load(best_model)
    model.load_state_dict(weights)
    x, y = random_sample(datasets['valid'])
    y_pred = model(x)
    valid_acc = accuracy(y_pred, y)
    title = f'Validation accuracy: {valid_acc:2.2%}'
    show(x.transpose(0, 2, 3, 1), subtitle=title, titles=y_pred)

    model_file = open(best_model, 'rb')
    model_proto = onnx_pb.ModelProto()
    model_proto.ParseFromString(model_file.read())
    coreml_model = convert(model_proto)


if __name__ == '__main__':
    main()
