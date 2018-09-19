from os.path import exists, isdir
from urllib.request import urlopen
from collections import defaultdict

import torch
import cv2 as cv
import numpy as np


def parse_annotations(arr):
    """
    Parses Pascal VOC dataset annotations into format suitable for next
    processing steps.
    """
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
    Converts Pascal bounding box from VOC into NumPy format.

    The original bounding boxes are represented as (x, y, w, h) tuples. The
    function converts these bounding boxes into (top, left, bottom, right)
    tuples, and switches x/y coordinates to make a converted array indexes
    consistent with Numpy.
    """
    x, y, w, h = bbox
    new_box = [y, x, y + h - 1, x + w - 1]
    return new_box


def to_voc(bbox):
    """
    Converts NumPy bounding boxes back into VOC format.

    The function performs an inverse transformation of the transformation
    performed with `from_voc` function.
    """
    top, left, bottom, right = bbox
    new_box = [left, top, right - left + 1, bottom - top + 1]
    return new_box


def open_image(path):
    """
    Opens an image using OpenCV given the file path.

    Args:
         path: A local file path or URL of the image.

    Return:
        image: The image in RGB format normalized to range between 0.0 - 1.0

    """
    flags = cv.IMREAD_UNCHANGED + cv.IMREAD_ANYDEPTH + cv.IMREAD_ANYCOLOR
    is_url = str(path).startswith('http')
    if not exists(path) and not is_url:
        raise OSError(f'No such file or directory: {path}')
    elif isdir(path) and not is_url:
        raise OSError(f'Is a directory: {path}')
    else:
        try:
            if is_url:
                r = urlopen(str(path))
                arr = np.asarray(bytearray(r.read()), dtype='uint8')
                image = cv.imdecode(arr, flags)
            else:
                image = cv.imread(str(path), flags)
            image = image.astype(np.float32)/255
            if image is None:
                raise OSError(f'File is not recognized by OpenCV: {path}')
        except Exception as e:
            raise OSError(f'Error handling image at: {path}') from e
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def read_sample(path, boxes, size=None):
    """
    Args:
        path: A local file path or URL of the image.
        boxes: An array with bounding boxes.
        size: An optional tuple or integer with the size used to rescale the
            read image. The image is rescaled without keeping aspect ratio.

    """
    image = open_image(path)
    old_size = image.shape[:2]
    if size is not None:
        size = (size, size) if isinstance(size, int) else tuple(size)
        image = cv.resize(image, size)
    new_size = image.shape[:2]
    if old_size != new_size:
        old_boxes = np.array(boxes)
        new_boxes = np.zeros_like(old_boxes)
        for i, box in enumerate(old_boxes.reshape(-1, 4)):
            box = resize_box(box, old_size, new_size)
            new_boxes[i*4:(i + 1)*4] = box
        boxes = new_boxes
    return image, boxes


def resize_box(box, old_size, new_size):
    y1, x1, y2, x2 = box
    old_h, old_w = old_size
    new_h, new_w = new_size
    h_ratio = new_h / float(old_h)
    w_ratio = new_w / float(old_w)
    new_box = [y1*h_ratio, x1*w_ratio, y2*h_ratio, x2*w_ratio]
    return new_box


def pad(arr, pad_value=0):
    longest = len(max(arr, key=len))
    padded = np.zeros((len(arr), longest), dtype=arr[0].dtype)
    for row, vec in enumerate(arr):
        n = len(vec)
        for i in range(longest):
            col = longest - i - 1
            padded[row, col] = pad_value if i >= n else vec[n - i - 1]
    return padded


def valid_box(vec):
    return np.count_nonzero(vec) >= 2


def t(obj, **kwargs):
    return torch.tensor(obj, **kwargs)


def to_np(*tensors):
    return [tensor.cpu().numpy() for tensor in tensors]


def hw2corners(centers, hw):
    """Converts an array of rectangles from (cx, cy, height, width)
    representations into (top, left, bottom, right) corners representation.
    """
    return torch.cat([centers - hw/2, centers + hw/2], dim=1)


def jaccard(a, b):
    intersection = intersect(a, b)
    union = area(a).unsqueeze(1) + area(b).unsqueeze(0) - intersection
    return intersection / union


def intersect(a, b):
    bottom_right = torch.min(a[:, None, 2:], b[None, :, 2:])
    top_left = torch.max(a[:, None, :2], b[None, :, :2])
    inter = torch.clamp((bottom_right - top_left), min=0)
    return torch.prod(inter, dim=2)


def area(box):
    h = box[:, 2] - box[:, 0]
    w = box[:, 3] - box[:, 1]
    return h * w


def make_grid(anchors=4, k=1):
    offset = 1/(anchors*2)
    points = np.linspace(offset, 1 - offset, anchors)
    anchors_x = np.repeat(points, anchors)
    anchors_y = np.tile(points, anchors)
    centers = np.stack([anchors_x, anchors_y], axis=1)
    sizes = np.array([(1/anchors, 1/anchors) for _ in range(anchors*anchors)])
    grid = np.c_[np.tile(centers, (k, 1)), np.tile(sizes, (k, 1))]
    return grid