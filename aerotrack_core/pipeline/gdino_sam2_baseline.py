"""Baseline 6: Grounding DINO (detector) + SAM2 (segmenter), hard-restart.

Architecturally identical to BL5 (YOLO-World + SAM2) but uses Grounding DINO
as the detector — a text-to-box open-vocabulary detector.
Detector logic is copied verbatim from GDinoSAM3BaselineTracker (BL4).
"""

import os

import torch

from aerotrack_core.models.groundingdino_sam3.groundingdino.util.inference import (
    Model as GroundingDINOModel,
)
from aerotrack_core.pipeline.gdino_sam3_baseline import _resolve_gdino_scale
from aerotrack_core.pipeline.sam2_tracker_base import BaseSAM2Tracker


class GDinoSAM2BaselineTracker(BaseSAM2Tracker):
    """BL6: Grounding DINO + SAM2 hard-restart tracker."""

    def __init__(
        self,
        sam2_checkpoint,
        gdino_checkpoint,
        gdino_config=None,
        gdino_bert_path=None,
        relocation_interval=30,
        box_threshold=0.3,
        text_threshold=0.25,
        max_dets=None,
        nms_iou_thr=0.5,
        match_iou_thr=0.3,
        max_mask_area_ratio=0.5,
        mask_nms_iou_thr=0.5,
        max_objects=20,
        infer_scale=1.0,
        use_track_lifecycle=True,
        lost_track_ttl_segments=2,
        lifecycle_match_score_thr=0.45,
        lifecycle_center_gate=2.5,
        lifecycle_area_ratio_min=0.35,
        lifecycle_area_ratio_max=2.8,
        verbose=False,
        show_progress=True,
    ):
        super().__init__(
            sam2_checkpoint=sam2_checkpoint,
            relocation_interval=relocation_interval,
            match_iou_thr=match_iou_thr,
            max_mask_area_ratio=max_mask_area_ratio,
            mask_nms_iou_thr=mask_nms_iou_thr,
            max_objects=max_objects,
            use_track_lifecycle=use_track_lifecycle,
            lost_track_ttl_segments=lost_track_ttl_segments,
            lifecycle_match_score_thr=lifecycle_match_score_thr,
            lifecycle_center_gate=lifecycle_center_gate,
            lifecycle_area_ratio_min=lifecycle_area_ratio_min,
            lifecycle_area_ratio_max=lifecycle_area_ratio_max,
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
        self.box_threshold = box_threshold
        # GroundingDINO still requires a numeric text_threshold for phrase parsing.
        self.text_threshold = 0.25 if text_threshold is None else text_threshold
        self.max_dets = max_dets
        self.nms_iou_thr = nms_iou_thr
        self.infer_scale = infer_scale
        self.gdino_model = None
        self.debug_gdino = self.verbose or (
            os.environ.get("AEROTRACK_DEBUG_GDINO", "0").lower() in ("1", "true", "yes")
        )

    def set_infer_scale(self, infer_scale):
        """Update GroundingDINO preprocessing scale without reloading weights."""
        try:
            infer_scale = float(infer_scale)
        except (TypeError, ValueError):
            return
        if abs(float(getattr(self, "infer_scale", 1.0)) - infer_scale) < 1e-6:
            return
        self.infer_scale = infer_scale
        if self.gdino_model is None:
            return
        short_side, max_size = _resolve_gdino_scale(self.infer_scale)
        self.gdino_model.set_resize(short_side=short_side, max_size=max_size)
        if short_side is not None:
            self._log(
                f"[GroundingDINO] infer_scale={self.infer_scale} -> "
                f"resize {short_side}/{max_size}"
            )

    # ------------------------------------------------------------------
    # NMS helpers (identical to BL4)
    # ------------------------------------------------------------------

    @staticmethod
    def _nms_torch(bboxes, scores, iou_thr):
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

    def _apply_nms(self, bboxes, scores, iou_thr=0.5):
        if bboxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=bboxes.device)
        try:
            from torchvision.ops import nms
            return nms(bboxes, scores, iou_thr)
        except Exception:
            return self._nms_torch(bboxes, scores, iou_thr)

    # ------------------------------------------------------------------
    # Detector loading (identical to BL4)
    # ------------------------------------------------------------------

    def _load_detector(self):
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

        short_side, max_size = _resolve_gdino_scale(self.infer_scale)
        self.gdino_model = GroundingDINOModel(
            model_config_path=self.gdino_config,
            model_checkpoint_path=self.gdino_checkpoint,
            device=self.device,
            short_side=short_side,
            max_size=max_size,
        )
        if short_side is not None:
            self._log(
                f"Grounding DINO loaded (infer_scale={self.infer_scale}, "
                f"resize {short_side}/{max_size})."
            )
        else:
            self._log("Grounding DINO loaded (default 800/1333 resize).")

    # ------------------------------------------------------------------
    # Detection (identical to BL4)
    # ------------------------------------------------------------------

    def detect_and_relocate(self, frame, text_prompt):
        detections, phrases = self.gdino_model.predict_with_caption(
            image=frame,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        if len(detections.xyxy) == 0:
            if self.debug_gdino:
                self._log(
                    f"[GDINO BL6] prompt='{text_prompt}' "
                    f"box_thr={self.box_threshold} text_thr={self.text_threshold} "
                    f"raw=0"
                )
            return [], []

        bboxes_np = detections.xyxy
        confs_np = detections.confidence
        raw_count = len(bboxes_np)
        raw_min = float(confs_np.min()) if raw_count else 0.0
        raw_max = float(confs_np.max()) if raw_count else 0.0

        # 1. First apply NMS
        bboxes_t = torch.from_numpy(bboxes_np).float()
        confs_t = torch.from_numpy(confs_np).float()
        keep = self._apply_nms(bboxes_t, confs_t, iou_thr=self.nms_iou_thr)
        bboxes_t = bboxes_t[keep]
        confs_t = confs_t[keep]
        nms_count = len(bboxes_t)

        # 2. Then apply top-k. If max_dets is not configured, fall back to
        # max_objects for backward compatibility with earlier GDINO baselines.
        max_dets_cfg = self.max_dets if self.max_dets is not None else self.max_objects
        max_dets = int(max_dets_cfg) if max_dets_cfg is not None else len(bboxes_t)
        if max_dets > 0 and len(bboxes_t) > max_dets:
            topk_idx = confs_t.topk(max_dets).indices
            bboxes_t = bboxes_t[topk_idx]
            confs_t = confs_t[topk_idx]
        final_count = len(bboxes_t)

        if self.debug_gdino:
            phrase_preview = phrases[:3] if phrases else []
            self._log(
                f"[GDINO BL6] prompt='{text_prompt}' "
                f"box_thr={self.box_threshold} text_thr={self.text_threshold} "
                f"raw={raw_count} nms={nms_count} final={final_count} "
                f"score_range=[{raw_min:.3f},{raw_max:.3f}] phrases={phrase_preview}"
            )

        if len(confs_t) > 1:
            order = confs_t.argsort(descending=True)
            bboxes_t = bboxes_t[order]
            confs_t = confs_t[order]

        ori_h, ori_w = frame.shape[:2]
        bboxes_t = bboxes_t.clamp(min=0)
        bboxes_t[:, 0::2] = bboxes_t[:, 0::2].clamp(max=ori_w)
        bboxes_t[:, 1::2] = bboxes_t[:, 1::2].clamp(max=ori_h)

        return bboxes_t.numpy().tolist(), confs_t.numpy().tolist()
