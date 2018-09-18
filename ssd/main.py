from pathlib import Path

import torch

from utils import to_np
from plots import VOCPlotter
from dataset import VOCDataset, VOCDataLoader


ROOT = Path.home().joinpath('data', 'voc2007')
TRAIN_JSON = ROOT / 'pascal_train2007.json'
TRAIN_JPEG = ROOT.joinpath('VOCdevkit', 'VOC2007', 'JPEGImages')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
    dataset = VOCDataset(TRAIN_JSON, TRAIN_JPEG, device=DEVICE)
    loader = VOCDataLoader(dataset, batch_size=12, num_workers=0)
    plotter = VOCPlotter(id2cat=dataset.id2cat, figsize=(12, 10))

    for batch in iter(loader):
        with plotter:
            plotter.plot_boxes(*to_np(*batch))
            break  # a single batch to verify everything works


if __name__ == '__main__':
    main()
