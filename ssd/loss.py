import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from utils import hw2corners, jaccard


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predictions, target):
        one_hot = self.one_hot_embedding(target)
        t_target = one_hot[:, :-1].contiguous()
        t_input = predictions[:, :-1]
        xe = F.binary_cross_entropy_with_logits(
            t_input, t_target, size_average=False)
        return xe / self.num_classes

    def one_hot_embedding(self, labels):
        return torch.eye(self.num_classes)[labels.data.cpu()]


def ssd_loss(y_pred, y_true, anchors, grid_sizes, loss_f, size=224):

    def filter_empty(boxes, classes):
        """Drops samples with boxes of zero width."""

        rescaled_boxes = boxes.view(-1, 4) / size
        index = (rescaled_boxes[:, 2] - rescaled_boxes[:, 0]) > 0
        [keep] = index.nonzero()
        return rescaled_boxes[keep], classes[keep]


    def activations_to_boxes(activations):
        """Converts activation values of top layers into bounding boxes."""

        tanh = F.tanh(activations)
        centers = (tanh[:, :2]/2 * grid_sizes) + anchors[:, :2]
        hw = (tanh[:, 2:]/2 + 1) * anchors[:, 2:]
        return hw2corners(centers, hw)


    def map_to_ground_truth(overlaps):
        """Converts an array with Jaccard metrics into predictions."""

        prior_overlap, prior_index = overlaps.max(1)
        gt_overlap, gt_index = overlaps.max(0)
        gt_overlap[prior_index] = 1.99
        for i, index in enumerate(prior_index):
            gt_index[index] = i
        return gt_overlap, gt_index


    box_loss, class_loss = 0, 0
    anchor_corners = hw2corners(anchors[:, :2], anchors[:, 2:])

    for pred_box_act, pred_cls, true_box, true_cls in zip(*y_pred, *y_true):
        boxes, classes = filter_empty(true_box, true_cls)
        pred_boxes = activations_to_boxes(pred_box_act)
        overlaps = jaccard(boxes.data, anchor_corners.data)
        gt_overlap, gt_index = map_to_ground_truth(overlaps)
        gt_classes = true_cls[gt_index]
        [positive] = np.nonzero(gt_overlap > 0.4)
        gt_boxes = true_box[gt_index]
        box_loss += (pred_boxes[positive] - gt_boxes[positive]).abs().mean()
        class_loss += loss_f(pred_cls, gt_classes)

    return box_loss + class_loss
