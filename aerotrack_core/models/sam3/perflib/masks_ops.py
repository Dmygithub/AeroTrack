# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch


def masks_to_boxes(masks: torch.Tensor, obj_ids: list[int]):
    with torch.autograd.profiler.record_function("perflib: masks_to_boxes"):
        # Sanity check based on callsite for replacement
        assert masks.shape[0] == len(obj_ids)
        assert masks.dim() == 3

        # Based on torchvision masks_to_boxes
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        N, H, W = masks.shape
        device = masks.device
        y = torch.arange(H, device=device).view(1, H)
        x = torch.arange(W, device=device).view(1, W)

        masks_with_obj = masks != 0  # N, H, W
        masks_with_obj_x = masks_with_obj.amax(
            dim=1
        )  # N, H (which columns have objects)
        masks_with_obj_y = masks_with_obj.amax(dim=2)  # N, W (which rows have objects)
        masks_without_obj_x = ~masks_with_obj_x
        masks_without_obj_y = ~masks_with_obj_y

        bounding_boxes_0 = torch.amin(
            (masks_without_obj_x * W) + (masks_with_obj_x * x), dim=1
        )
        bounding_boxes_1 = torch.amin(
            (masks_without_obj_y * H) + (masks_with_obj_y * y), dim=1
        )
        bounding_boxes_2 = torch.amax(masks_with_obj_x * x, dim=1)
        bounding_boxes_3 = torch.amax(masks_with_obj_y * y, dim=1)

        bounding_boxes = torch.stack(
            [bounding_boxes_0, bounding_boxes_1, bounding_boxes_2, bounding_boxes_3],
            dim=1,
        ).to(dtype=torch.float)
        assert bounding_boxes.shape == (N, 4)
        assert bounding_boxes.device == masks.device
        assert bounding_boxes.dtype == torch.float
        return bounding_boxes


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    """
    Compute the IoU (Intersection over Union) between predicted masks and ground truth masks.

    We use a chunked loop over `pred_masks` rows instead of broadcasting the full
    (N, M, H*W) boolean tensor at once.  For large N/M or high-resolution frames
    (e.g. 1080×1920) the naive broadcast can require tens of GiB and triggers OOM.
    The chunked version keeps peak VRAM at O(chunk_size × M × H*W).

    Args:
      - pred_masks:  (N, H, W) bool Tensor, binary predicted segmentation masks
      - gt_masks:    (M, H, W) bool Tensor, binary ground truth segmentation masks
      - chunk_size:  number of pred rows processed at a time (trade-off: smaller = less VRAM)
    Returns:
      - ious: (N, M) float Tensor, IoU for every (pred, gt) pair
    """
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    N, H, W = pred_masks.shape
    M = gt_masks.shape[0]
    HW = H * W

    # Pre-flatten gt once: (M, H*W)
    gt_flat = gt_masks.view(M, HW)
    gt_areas = gt_flat.sum(dim=1).float()  # (M,)

    ious = torch.empty(N, M, device=pred_masks.device, dtype=torch.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        # pred chunk: (chunk, H*W)
        pred_chunk = pred_masks[start:end].view(end - start, HW)
        pred_areas = pred_chunk.sum(dim=1).float()  # (chunk,)

        # intersection: (chunk, M)  — peak tensor is chunk × M × HW bits (bool)
        inter = torch.mm(pred_chunk.float(), gt_flat.float().t())  # float mul avoids bool broadcast
        # union = pred_area + gt_area - intersection
        union = pred_areas.unsqueeze(1) + gt_areas.unsqueeze(0) - inter
        ious[start:end] = inter / union.clamp(min=1)

    return ious  # shape: (N, M)
