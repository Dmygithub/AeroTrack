import pycocotools.mask as mask_util
import numpy as np


class PGIoUCalculator:
    """Prompt-Guided IoU (PG-IoU) for mask quality under GT box guidance."""

    def __init__(self):
        self.ious = []

    def add_mask_pair(self, pred_mask: np.ndarray, gt_mask: np.ndarray):
        """Accumulate IoU for one aligned pred/GT mask pair.

        Args:
            pred_mask: (H, W) bool/uint8 array or RLE dict.
            gt_mask: (H, W) bool/uint8 array or RLE dict.
        """
        if isinstance(pred_mask, dict) and 'counts' in pred_mask:
            pred_rle = pred_mask
        else:
            pred_rle = mask_util.encode(np.asfortranarray(pred_mask.astype(np.uint8)))

        if isinstance(gt_mask, dict) and 'counts' in gt_mask:
            gt_rle = gt_mask
        else:
            gt_rle = mask_util.encode(np.asfortranarray(gt_mask.astype(np.uint8)))

        iou = mask_util.iou([pred_rle], [gt_rle], [0])[0][0]
        self.ious.append(iou)

    def get_mean_iou(self) -> float:
        """Return mean IoU over all accumulated mask pairs."""
        if not self.ious:
            return 0.0
        return float(np.mean(self.ious))

    def report(self) -> str:
        """Return a human-readable PG-IoU summary string."""
        mean_iou = self.get_mean_iou()
        return f"PG-IoU (Prompt-Guided-IoU): {mean_iou*100:.2f}% (over {len(self.ious)} mask pairs)"
