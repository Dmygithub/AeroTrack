"""Baseline 7: Incremental SAM3 + SAM3 tracker.

Uses SAM3's internal text-driven detector at key frames.  Inherits the
warm-restart incremental architecture from ``IncrementalOVTracker``.

Detection strategy: at each key frame (after ``reset_session`` empties the
session), call ``add_prompt`` with the text query on the main session.
The response directly contains ``out_binary_masks`` and ``out_obj_ids``;
bounding boxes are computed from these masks.  The session is then reset
again to clear temporary objects so the base class can re-inject old tracks
and inject new ones cleanly.
"""

import numpy as np

from aerotrack_core.pipeline.incremental_tracker_base import IncrementalOVTracker


class SAM3IncrementalTracker(IncrementalOVTracker):
    """BL7: SAM3 text detection + SAM3 tracker.

    At each key frame the session is empty (just reset by the base class).
    ``add_prompt`` performs text-based object detection and returns masks
    directly in its response.  We extract bounding boxes from these masks,
    then reset the session to clear the temporary objects.
    """

    def __init__(
        self,
        sam3_checkpoint,
        relocation_interval=30,
        match_iou_thr=0.3,
        max_objects=20,
        new_det_thresh=0.85,
        max_trk_keep_alive=8,
        max_missing_segments=1,
        max_mask_area_ratio=0.5,
        verbose=False,
        show_progress=True,
    ):
        super().__init__(
            sam3_checkpoint,
            max_objects=max_objects,
            new_det_thresh=new_det_thresh,
            max_trk_keep_alive=max_trk_keep_alive,
            match_iou_thr=match_iou_thr,
            max_missing_segments=max_missing_segments,
            max_mask_area_ratio=max_mask_area_ratio,
            verbose=verbose,
            show_progress=show_progress,
        )
        self.relocation_interval = relocation_interval

    def _load_detector(self):
        pass

    def detect_and_relocate(self, frame, text_prompt):
        return None, None

    def detect_on_keyframe(self, frame, frame_idx, text_prompt, inference_state):
        """Run SAM3 text detection via ``add_prompt`` on the main session.

        The session is guaranteed to be empty (just reset by the base class).
        ``add_prompt`` responses contain ``out_binary_masks`` and ``out_obj_ids``.
        After bbox extraction the session is reset to clean up temporary objects.

        Returns:
            bboxes: list of [x1, y1, x2, y2] pixel coordinates.
            confidences: list of float scores.
        """
        session_id = None
        for sid, entry in self.video_predictor._ALL_INFERENCE_STATES.items():
            if entry.get("state") is inference_state:
                session_id = sid
                break
        if session_id is None:
            self._log("[BL7] Cannot find session_id; skipping detection.")
            return [], []

        try:
            result = self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    text=text_prompt,
                )
            )

            outputs = result.get("outputs")
            if not outputs:
                return [], []

            masks = outputs.get("out_binary_masks")
            obj_ids = outputs.get("out_obj_ids", [])
            probs = outputs.get("out_probs")

            # Fallback to out_boxes_xywh if masks unavailable.
            if masks is None:
                out_boxes = outputs.get("out_boxes_xywh")
                if out_boxes is None:
                    return [], []
                if hasattr(out_boxes, "cpu"):
                    out_boxes = out_boxes.cpu().numpy()
                if probs is not None and hasattr(probs, "cpu"):
                    probs = probs.cpu().numpy()
                H, W = frame.shape[:2]
                bboxes, confs = [], []
                for i in range(len(out_boxes)):
                    cx, cy, w, h = out_boxes[i]
                    x1 = int((cx - w / 2) * W)
                    y1 = int((cy - h / 2) * H)
                    x2 = int((cx + w / 2) * W)
                    y2 = int((cy + h / 2) * H)
                    bboxes.append([x1, y1, x2, y2])
                    confs.append(float(probs[i]) if probs is not None and i < len(probs) else 1.0)
                self._log(f"[BL7] Text detection found {len(bboxes)} objects (from boxes).")
                self.video_predictor.handle_request(
                    request=dict(type="reset_session", session_id=session_id)
                )
                return bboxes, confs

            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            if probs is not None and hasattr(probs, "cpu"):
                probs = probs.cpu().numpy()

            if len(masks) == 0:
                return [], []

            bboxes, confs = [], []
            for i in range(len(masks)):
                m = masks[i]
                if m.ndim == 3:
                    m = m.squeeze(0)
                m_bool = m.astype(bool) if m.dtype != bool else m
                bbox = self._mask_to_bbox(m_bool)
                if bbox is not None:
                    bboxes.append(bbox)
                    confs.append(float(probs[i]) if probs is not None and i < len(probs) else 1.0)

            self._log(f"[BL7] Text detection found {len(bboxes)} objects.")

            # Clean up temporary objects from this detection pass.
            self.video_predictor.handle_request(
                request=dict(type="reset_session", session_id=session_id)
            )

            return bboxes, confs

        except Exception as e:
            self._log(f"[BL7] Text detection failed: {e}")
            import traceback
            traceback.print_exc()
            return [], []
