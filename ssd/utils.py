from os.path import exists, isdir
from urllib.request import urlopen
from collections import defaultdict

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
    return np.array(new_box)


def to_voc(bbox):
    """
    Converts NumPy bounding boxes back into VOC format.

    The function performs an inverse transformation of the transformation
    performed with `from_voc` function.
    """
    top, left, bottom, right = bbox
    new_box = [left, top, right - left + 1, bottom - top + 1]
    return np.array(new_box)


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
            return cv.cvtColor(image, cv.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError(f'Error handling image at: {path}') from e
