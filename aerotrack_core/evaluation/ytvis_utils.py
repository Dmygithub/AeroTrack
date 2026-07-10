"""Convert tracker NPZ cache directories into YTVIS-format predictions."""

import os
import cv2
import numpy as np
import pycocotools.mask as mask_util


def cache_to_ytvis_predictions(
    cache_dir: str,
    video_id: int,
    category_id: int,
    num_frames: int,
    H: int,
    W: int,
) -> list:
    """Read per-frame NPZ caches and build YTVIS track-level predictions.

    Returns:
        List of dicts with video_id, category_id, segmentations, score,
        bboxes, and areas per track.
    """
    per_track: dict = {}

    for frame_idx in range(num_frames):
        npz_path = os.path.join(cache_dir, f"{frame_idx:06d}.npz")
        if not os.path.exists(npz_path):
            continue

        with np.load(npz_path) as data:
            masks = data["masks"]
            track_ids = data["track_ids"]
            frame_scores = data["scores"] if "scores" in data.files else None

        if masks.size == 0 or track_ids.size == 0:
            continue

        for i, tid in enumerate(track_ids):
            tid = int(tid)
            mask = masks[i]
            if mask.ndim == 3:
                mask = mask[0]

            if mask.shape != (H, W):
                mask = cv2.resize(
                    mask.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            if tid not in per_track:
                per_track[tid] = {
                    "segmentations": [None] * num_frames,
                    "bboxes":        [None] * num_frames,
                    "areas":         [None] * num_frames,
                    "scores":        [],
                }

            m_fortran = np.asfortranarray(mask.astype(np.uint8))
            rle = mask_util.encode(m_fortran)
            bbox = mask_util.toBbox(rle).tolist()
            rle["counts"] = rle["counts"].decode("ascii")
            area = int(mask.sum())

            per_track[tid]["segmentations"][frame_idx] = rle
            per_track[tid]["bboxes"][frame_idx] = bbox
            per_track[tid]["areas"][frame_idx] = area
            if frame_scores is not None and i < len(frame_scores):
                score = float(frame_scores[i])
                if np.isfinite(score):
                    per_track[tid]["scores"].append(float(np.clip(score, 0.0, 1.0)))

    predictions = []
    for tid, track_data in per_track.items():
        # Track-level score for AP-style metrics. Detector-based baselines
        # write per-frame detector confidences into cache; combine their mean
        # confidence with temporal coverage. Older caches and SAM3-native BL2
        # do not have detector scores, so they keep the original coverage score.
        valid_frames = sum(1 for s in track_data["segmentations"] if s is not None)
        coverage = valid_frames / max(1, num_frames)
        if track_data["scores"]:
            det_score = float(np.mean(track_data["scores"]))
            score = det_score * float(np.sqrt(coverage))
        else:
            score = coverage
        score = float(np.clip(score, 0.0, 1.0))

        predictions.append({
            "video_id":      video_id,
            "category_id":   category_id,
            "segmentations": track_data["segmentations"],
            "bboxes":        [b if b is not None else [0, 0, 0, 0] for b in track_data["bboxes"]],
            "areas":         [a if a is not None else 0 for a in track_data["areas"]],
            "score":         score,
        })

    return predictions
