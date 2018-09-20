import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from misc import hw2corners, jaccard


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predictions, target):
        one_hot = self.one_hot_embedding(target)
        t_target = one_hot[:, :-1].contiguous()
        t_input = predictions[:, :-1]
        xe = F.binary_cross_entropy_with_logits(
            t_input, t_target, reduction='sum')
        return xe / self.num_classes

    def one_hot_embedding(self, labels):
        device = labels.device
        matrix = torch.eye(self.num_classes + 1)[labels.data.cpu()]
        return matrix.to(device)


def ssd_loss(y_pred, y_true, anchors, grid_sizes, loss_f, n_classes, size=224):

    def get_relevant(boxes, classes):
        """Drops samples with boxes of zero width."""

        boxes = boxes.view(-1, 4).float() / size
        index = (boxes[:, 2] - boxes[:, 0]) > 0
        keep = index.nonzero()[:, 0]
        return boxes[keep], classes[keep]


    def activations_to_boxes(activations):
        """Converts activation values of top layers into bounding boxes."""

        tanh = torch.tanh(activations)
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


    anchor_corners = hw2corners(anchors[:, :2], anchors[:, 2:])

    box_loss, class_loss = 0, 0
    for pred_bb, pred_cls, true_bb, true_cls in zip(*y_pred, *y_true):
        true_bb, true_cls = get_relevant(true_bb, true_cls)
        activ_bb = activations_to_boxes(pred_bb)
        overlaps = jaccard(true_bb.data, anchor_corners.data)
        gt_overlap, gt_index = map_to_ground_truth(overlaps)
        gt_class = true_cls[gt_index]
        pos = gt_overlap > 0.4
        pos_index = torch.nonzero(pos)[:, 0]
        gt_class[1 - pos] = n_classes
        gt_bb = true_bb[gt_index]
        box_loss += (activ_bb[pos_index] - gt_bb[pos_index]).abs().mean()
        class_loss += loss_f(pred_cls, gt_class)

    return box_loss + class_loss
