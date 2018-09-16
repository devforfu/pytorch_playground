from collections import defaultdict

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
