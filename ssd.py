import os
import json
import urllib
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects

import torch
from torch import optim

from fastai.conv_learner import ConvLearner, resnet34
from fastai.conv_learner import tfms_from_model, CropType
from fastai.dataset import ImageClassifierData, to_np


PATH = Path.home()/'data'/'voc2007'
IMAGES, ANNOTATIONS, CATEGORIES = 'images', 'annotations', 'categories'
FILE_NAME, ID, BBOX = 'file_name', 'id', 'bbox'
IMG_ID, CAT_ID = 'image_id', 'category_id'


def parse_annotations(arr):
    parsed = defaultdict(list)
    for annot in arr:
        if annot['ignore']:
            continue
        bbox_raw = annot['bbox']
        bbox_hw = from_voc(bbox_raw)
        parsed[annot['image_id']].append((bbox_hw, annot['category_id']))
    return dict(parsed)


def from_voc(bbox):
    """
    Converts Pascal bounding box from (x, y, w, h) into
    (top, left, bottom, right) format, and switch x/y coordinates to
    make a converted array indexes consistent with Numpy.
    """
    x, y, w, h = bbox
    new_box = [y, x, y + h - 1, x + w - 1]
    return np.array(new_box)


def to_voc(bbox):
    top, left, bottom, right = bbox
    new_box = [left, top, right - left + 1, bottom - top + 1]
    return np.array(new_box)


def show_img(im, figsize=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.set_xticks(np.linspace(0, 224, 8))
    ax.set_yticks(np.linspace(0, 224, 8))
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax


def draw_outline(obj, lw):
    effects = [
        patheffects.Stroke(linewidth=lw, foreground='black'),
        patheffects.Normal()]
    obj.set_path_effects(effects)


def draw_rect(ax, bbox, color='white'):
    rect = patches.Rectangle(
        bbox[:2], *bbox[-2:],
        fill=False, edgecolor=color, lw=2)
    patch = ax.add_patch(rect)
    draw_outline(patch, 4)


def draw_text(ax, xy, text, size=14, color='white'):
    text = ax.text(
        *xy, text,
        va='top', color=color,
        fontsize=size, weight='bold')
    draw_outline(text, 1)


def draw_image(img, ann, categories):
    ax = show_img(img, figsize=(8, 6))
    for bbox, cat in ann:
        bbox = to_voc(bbox)
        draw_rect(ax, bbox)
        draw_text(ax, bbox[:2], categories[cat], size=16)


class Drawer:

    def __init__(self, root, annotations, files, categories):
        self.root = root
        self.annotations = annotations
        self.files = files
        self.categories = categories

    def draw(self, index):
        annotation = self.annotations[index]
        image = open_image(self.root / self.files[index])
        draw_image(image, annotation, self.categories)
        plt.pause(0.001)


def open_image(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in RGB format as numpy array of floats normalized
        to range between 0.0 - 1.0

    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn) and not str(fn).startswith("http"):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn) and not str(fn).startswith("http"):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        try:
            if str(fn).startswith("http"):
                req = urllib.urlopen(str(fn))
                image = np.asarray(bytearray(req.read()), dtype="uint8")
                im = cv2.imdecode(image, flags).astype(np.float32)/255
            else:
                im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None:
                raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


def main():
    with open(PATH/'pascal_train2007.json') as file:
        train_json = json.load(file)

    categories = {obj[ID]: obj['name'] for obj in train_json[CATEGORIES]}
    train_files = {obj[ID]: obj[FILE_NAME] for obj in train_json[IMAGES]}
    train_indexes = [obj[ID] for obj in train_json[IMAGES]]
    train_annotations = parse_annotations(train_json[ANNOTATIONS])

    JPEGS = 'VOCdevkit/VOC2007/JPEGImages'
    drawer = Drawer(PATH / JPEGS, train_annotations, train_files, categories)
    # drawer.draw(12)

    factory = resnet34
    batch_size = 64
    CSV = PATH / 'tmp' / 'mc.csv'

    transforms = tfms_from_model(factory, 224, crop_type=CropType.NO)
    data = ImageClassifierData.from_csv(
        PATH, JPEGS, CSV, tfms=transforms, bs=batch_size)

    learner = ConvLearner.pretrained(factory, data)
    learner.opt_fn = optim.Adam

    lr = 2e-2
    learner.fit(lr, 1, cycle_len=3, use_clr=(32, 5))

    # lrs = np.array([lr/100, lr/10, lr])
    # learner.freeze_to(-2)
    # learner.fit(lrs/10, 1, cycle_len=5, use_clr=(32, 5))
    #
    y = learner.predict()
    x, _ = next(iter(data.val_dl))
    x = to_np(x)
    images = data.val_ds.denorm(x)

    fig, axes = plt.subplots(3, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        image = images[i]
        [non_zero] = np.nonzero(y[i] > 0.4)
        classes = '\n'.join([data.classes[index] for index in non_zero])
        ax = show_img(image, ax=ax)
        draw_text(ax, (0, 0), classes)
    plt.tight_layout()
    plt.pause(0.001)


if __name__ == '__main__':
    main()
