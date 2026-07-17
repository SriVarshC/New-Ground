"""Segmentation metrics shared across all phases.

IoU is implemented by hand rather than pulled from a library -- a good
interview talking point, and simple enough that hand-rolling it teaches
the metric better than importing it opaquely.
"""
from __future__ import annotations

import numpy as np
import torch


def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    """Intersection-over-union per class.

    IoU = (pixels where pred AND target both say this class)
          / (pixels where pred OR target says this class)

    Returns an array of shape (num_classes,). A class gets NaN if it's
    absent from both prediction and target in this batch (nothing to
    divide -- avoids a misleading 0 or 1).
    """
    ious = np.full(num_classes, np.nan, dtype=np.float64)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    for c in range(num_classes):
        pred_c = pred_flat == c
        target_c = target_flat == c
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        intersection = (pred_c & target_c).sum().item()
        ious[c] = intersection / union

    return ious