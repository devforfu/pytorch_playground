import math
from pathlib import Path

import torch
from torch import optim

from misc import to_np, make_grid, hw2corners, t
from plots import VOCPlotter
from models import SSD
from loss import ssd_loss, BinaryCrossEntropyLoss
from dataset import VOCDataset, VOCDataLoader
from core.loop import Loop
from core.schedule import CosineAnnealingLR
from core.callbacks import Logger


ROOT = Path.home().joinpath('data', 'voc2007')
TRAIN_JSON = ROOT / 'pascal_train2007.json'
TRAIN_JPEG = ROOT.joinpath('VOCdevkit', 'VOC2007', 'JPEGImages')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
    bs = 64
    n_anchors = 4
    dataset = VOCDataset(TRAIN_JSON, TRAIN_JPEG, device=DEVICE)
    loader = VOCDataLoader(dataset, batch_size=bs, num_workers=0)
    # plotter = VOCPlotter(id2cat=dataset.id2cat, figsize=(12, 10))
    #
    # for images, (boxes, classes) in iter(loader):
    #     with plotter:
    #         plotter.plot_boxes(*to_np(images, boxes, classes))
    #         break  # a single batch to verify everything works

    n_classes = len(dataset.id2cat)
    cycle_len = math.ceil(len(dataset)/bs)
    model = SSD(n_classes=n_classes, bias=-3.)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = CosineAnnealingLR(optimizer, t_max=cycle_len)
    loop = Loop(model, optimizer, scheduler, device=DEVICE)

    anchors, grid_sizes = [
        x.to(DEVICE) for x in (
            t(make_grid(n_anchors), requires_grad=False).float(),
            t([1/n_anchors], requires_grad=False).unsqueeze(1))]

    bce_loss = BinaryCrossEntropyLoss(n_classes)
    loss_fn = lambda x, y: ssd_loss(x, y, anchors, grid_sizes, bce_loss, n_classes)

    loop.run(
        train_data=loader,
        epochs=100,
        loss_fn=loss_fn,
        callbacks=[Logger()]
    )



if __name__ == '__main__':
    main()
