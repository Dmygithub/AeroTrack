"""Baseline 8: Incremental Grounding DINO + SAM3 tracker.

Uses Grounding DINO as the external open-vocabulary detector at key frames.
Inherits the incremental single-session architecture from ``IncrementalOVTracker``.
Grounding DINO loading and inference logic is ported from ``GDinoSAM3BaselineTracker`` (BL4).
"""

import os
import torch
import numpy as np

from aerotrack_core.pipeline.incremental_tracker_base import IncrementalOVTracker
from aerotrack_core.models.groundingdino_sam3.groundingdino.util.inference import (
    Model as GroundingDINOModel,
)


class GDinoIncrementalTracker(IncrementalOVTracker):
    """BL8: Grounding DINO (detector) + SAM3 (segmenter), incremental mode.

    Identical incremental architecture to BL7/BL9 but uses Grounding DINO instead of
    SAM3 text detector / YOLO-World at key frames.
    """

    def __init__(
        self,
        sam3_checkpoint,
        gdino_checkpoint,
        gdino_config=None,
        gdino_bert_path=None,
        relocation_interval=30,
        match_iou_thr=0.3,
        box_threshold=0.3,
        text_threshold=0.25,
        max_dets=None,
        nms_iou_thr=0.5,
        max_objects=20,
        new_det_thresh=0.85,
        max_trk_keep_alive=8,
        max_missing_segments=1,
        max_mask_area_ratio=0.5,
        sam3_tuning_cfg=None,
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
        self.gdino_checkpoint = gdino_checkpoint
        if gdino_config is None:
            self.gdino_config = os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                "../models/groundingdino_sam3/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            ))
        else:
            self.gdino_config = gdino_config

        self.gdino_bert_path = gdino_bert_path
        self.relocation_interval = relocation_interval
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_dets = max_dets
        self.nms_iou_thr = nms_iou_thr
        self.gdino_model = None
        self._sam3_tuning_cfg = sam3_tuning_cfg or {}

    # NMS helpers

    @staticmethod
    def _nms_torch(bboxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
        if bboxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=bboxes.device)
        boxes = bboxes.float()
        scores = scores.float()
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
            order = rest[iou <= iou_thr]
        if len(keep) == 0:
            return torch.empty((0,), dtype=torch.long, device=bboxes.device)
        return torch.stack(keep)

    def _apply_nms(self, bboxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
        if bboxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=bboxes.device)
        try:
            from torchvision.ops import nms
            return nms(bboxes, scores, iou_thr)
        except Exception:
            return self._nms_torch(bboxes, scores, iou_thr)

    def _load_detector(self):
        """Load Grounding DINO with optional offline BERT tokenizer path."""
        self._log(f"Loading Grounding DINO from {self.gdino_checkpoint}...")

        if self.gdino_bert_path and os.path.isdir(self.gdino_bert_path):
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            self._log(f"Using local BERT path: {self.gdino_bert_path}")

            import aerotrack_core.models.groundingdino_sam3.groundingdino.util.get_tokenlizer as _gt
            _original_get_tokenlizer = _gt.get_tokenlizer
            _local_bert = os.path.abspath(self.gdino_bert_path)

            def _patched_get_tokenlizer(text_encoder_type):
                if text_encoder_type == "bert-base-uncased":
                    text_encoder_type = _local_bert
                return _original_get_tokenlizer(text_encoder_type)

            _gt.get_tokenlizer = _patched_get_tokenlizer

            _original_get_plm = _gt.get_pretrained_language_model

            def _patched_get_plm(text_encoder_type):
                if text_encoder_type == "bert-base-uncased":
                    text_encoder_type = _local_bert
                return _original_get_plm(text_encoder_type)

            _gt.get_pretrained_language_model = _patched_get_plm

        self.gdino_model = GroundingDINOModel(
            model_config_path=self.gdino_config,
            model_checkpoint_path=self.gdino_checkpoint,
            device=self.device,
        )
        self._log("Grounding DINO loaded.")

    # ------------------------------------------------------------------
    # SAM3 tuning
    # ------------------------------------------------------------------

    def load_models(self):
        super().load_models()
        model = self.video_predictor.model
        tuning = self._sam3_tuning_cfg
        if tuning.get("disable_hotstart", False):
            model._warm_up_complete = False
            model.hotstart_delay = 0
        if tuning.get("hotstart_delay") is not None:
            model.hotstart_delay = int(tuning["hotstart_delay"])
        if tuning.get("score_threshold_detection") is not None:
            model.score_threshold_detection = float(tuning["score_threshold_detection"])
        tracker = getattr(model, "tracker", None)
        if tracker is not None:
            if tuning.get("num_maskmem") is not None:
                tracker.num_maskmem = int(tuning["num_maskmem"])
            if tuning.get("max_obj_ptrs_in_encoder") is not None:
                tracker.max_obj_ptrs_in_encoder = int(tuning["max_obj_ptrs_in_encoder"])

    # ------------------------------------------------------------------
    # IncrementalOVTracker abstract method
    # ------------------------------------------------------------------

    def detect_and_relocate(self, frame, text_prompt):
        # Not used in incremental path; kept for interface compat.
        return None, None

    def detect_on_keyframe(self, frame, frame_idx, text_prompt, inference_state):
        """Run Grounding DINO detection on one frame.
        
        Args:
            frame: BGR numpy image (H, W, 3).
            frame_idx: Frame index.
            text_prompt: Open-vocabulary category text.
            inference_state: Current SAM3 inference state.
            
        Returns:
            bboxes: list of [x1, y1, x2, y2] pixel coordinates.
            confs: list of float scores.
        """
        detections, _phrases = self.gdino_model.predict_with_caption(
            image=frame,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        if len(detections.xyxy) == 0:
            return [], []

        bboxes_np = detections.xyxy  # (N, 4) xyxy
        confs_np = detections.confidence  # (N,)

        # 1. First apply NMS
        bboxes_t = torch.from_numpy(bboxes_np).float()
        confs_t = torch.from_numpy(confs_np).float()
        keep = self._apply_nms(bboxes_t, confs_t, iou_thr=self.nms_iou_thr)
        bboxes_t = bboxes_t[keep]
        confs_t = confs_t[keep]

        # 2. Then apply top-k. If max_dets is not configured, fall back to
        # max_objects for backward compatibility with earlier GDINO baselines.
        max_dets_cfg = self.max_dets if self.max_dets is not None else self.max_objects
        max_dets = int(max_dets_cfg) if max_dets_cfg is not None else len(bboxes_t)
        if max_dets > 0 and len(bboxes_t) > max_dets:
            topk_idx = confs_t.topk(max_dets).indices
            bboxes_t = bboxes_t[topk_idx]
            confs_t = confs_t[topk_idx]

        if len(confs_t) > 1:
            order = confs_t.argsort(descending=True)
            bboxes_t = bboxes_t[order]
            confs_t = confs_t[order]

        ori_h, ori_w = frame.shape[:2]
        bboxes_t = bboxes_t.clamp(min=0)
        bboxes_t[:, 0::2] = bboxes_t[:, 0::2].clamp(max=ori_w)
        bboxes_t[:, 1::2] = bboxes_t[:, 1::2].clamp(max=ori_h)

        return bboxes_t.numpy().tolist(), confs_t.numpy().tolist()
