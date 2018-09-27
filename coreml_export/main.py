"""
An example of converting PyTorch model into CoreML format using ONNX as an
intermediate format.
"""
import math
from pathlib import Path

from onnx_coreml import convert
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

import torch
from torch import optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.loop import Loop
from core.metrics import accuracy
from core.callbacks import default_callbacks

from model import ResNet


DATA_ROOT = Path.home() / 'data' / 'emnist'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
STATS = [0.17325], [0.33163]


def main():
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

    n_epochs = 3
    batch_size = 4096
    num_workers = 0  # cpu_count()

    datasets = load_dataset(data_transforms, batch_size, num_workers)
    n_samples = len(datasets['train']['loader'])
    n_batches = math.ceil(n_samples / batch_size)

    model = ResNet(10)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    sched = CosineAnnealingLR(opt, T_max=n_batches/4, eta_min=1e-5)
    loop = Loop(model, opt, sched, device=DEVICE)

    # loop.run(train_data=datasets['train']['loader'],
    #          valid_data=datasets['valid']['loader'],
    #          loss_fn=F.cross_entropy,
    #          metrics=[accuracy],
    #          callbacks=default_callbacks(),
    #          epochs=n_epochs)

    # file_name = loop['Checkpoint'].best_model
    dataset = datasets['valid']['loader']
    # validate_model(model, file_name, dataset, DEVICE)
    export_to_core_ml(model)


def load_dataset(data_transforms, batch_size=1024, num_workers=0,
                 root=DATA_ROOT, split='digits'):

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


def show_predictions(images, suptitle='', titles=None, dims=(4, 4),
                     figsize=(12, 12), stats=STATS):

    f, ax = plt.subplots(*dims, figsize=figsize)
    titles = titles or []
    f.suptitle(suptitle)
    mean, std = stats or [0, 1]
    images *= std
    images += mean
    for i, (img, ax) in enumerate(zip(images, ax.flat)):
        ax.imshow(img.reshape(28, 28))
        if i < len(titles):
            ax.set_title(titles[i])
    plt.show()


def validate_model(model, model_file, dataset, device):
    weights = torch.load(model_file)
    model.load_state_dict(weights)
    x, y = [t.to(device) for t in random_sample(dataset)]
    y_pred = model(x)
    valid_acc = accuracy(y_pred, y)
    title = f'Validation accuracy: {valid_acc:2.2%}'
    titles = [str(x) for x in to_np(y_pred.argmax(dim=1))]
    images = to_np(x.permute(0, 3, 2, 1))
    show_predictions(images, title, titles)


def random_sample(dataset, n=16):
    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    return next(iter(loader))


def export_to_core_ml(model):
    model.eval()
    device = model.fc.weight.device
    dummy_input = torch.randn(16, 1, 28, 28, requires_grad=True).to(device)
    torch.onnx.export(model, dummy_input, 'model.onnx', export_params=True)
    core_ml_model = convert('model.onnx')
    core_ml_model.save('model.mlmodel')


def to_np(*tensors):

    def convert_to_numpy(obj):
        return obj.detach().cpu().numpy()

    if len(tensors) == 1:
        return convert_to_numpy(tensors[0])
    return [convert_to_numpy(tensor) for tensor in tensors]


if __name__ == '__main__':
    main()
