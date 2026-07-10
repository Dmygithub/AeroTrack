import numpy as np


class ODRCalculator:
    """Open-Detection-Recall: GT recall when pred box center falls inside GT box."""

    def __init__(self):
        self.total_gt = 0
        self.total_tp = 0
        self.cat_stats = {}

    def add_frame_results(self, pred_boxes: list, gt_boxes: list, pred_labels: list, gt_labels: list):
        """Add one frame of box predictions and ground truth.

        Args:
            pred_boxes: List of [x1, y1, x2, y2] boxes.
            gt_boxes: List of [x1, y1, x2, y2] boxes.
            pred_labels: Category id per predicted box.
            gt_labels: Category id per GT box.
        """
        pred_boxes = np.array(pred_boxes) if len(pred_boxes) > 0 else np.empty((0, 4))
        gt_boxes = np.array(gt_boxes) if len(gt_boxes) > 0 else np.empty((0, 4))
        pred_labels = np.array(pred_labels)
        gt_labels = np.array(gt_labels)

        for glabel in gt_labels:
            if glabel not in self.cat_stats:
                self.cat_stats[glabel] = {"gt": 0, "tp": 0}
            self.cat_stats[glabel]["gt"] += 1

        self.total_gt += len(gt_boxes)

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return

        pred_centers_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
        pred_centers_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0

        matched_gt_indices = set()

        for i in range(len(pred_boxes)):
            cx, cy = pred_centers_x[i], pred_centers_y[i]
            plab = pred_labels[i]

            for j in range(len(gt_boxes)):
                if j in matched_gt_indices:
                    continue
                if gt_labels[j] != plab:
                    continue

                gt_x1, gt_y1, gt_x2, gt_y2 = gt_boxes[j]

                if gt_x1 <= cx <= gt_x2 and gt_y1 <= cy <= gt_y2:
                    matched_gt_indices.add(j)
                    self.total_tp += 1
                    if plab not in self.cat_stats:
                        self.cat_stats[plab] = {"gt": 0, "tp": 0}
                    self.cat_stats[plab]["tp"] += 1
                    break

    def get_odr(self) -> float:
        """Return aggregate open-detection recall."""
        if self.total_gt == 0:
            return 0.0
        return self.total_tp / self.total_gt

    def report(self) -> str:
        """Return a human-readable ODR summary string."""
        odr = self.get_odr()
        return f"ODR (Open-Detection-Recall): {odr*100:.2f}% (TP: {self.total_tp} / GT: {self.total_gt})"
