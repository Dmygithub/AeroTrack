import os
import torch
import numpy as np

from aerotrack_core.pipeline.ov_tracker_base import BaseOVTracker
from aerotrack_core.models.groundingdino_sam3.groundingdino.util.inference import (
    Model as GroundingDINOModel,
)


# GroundingDINO SwinT uses patch_size=4; round resolution to a multiple of 4
# to stay aligned with the patch grid and avoid fractional padding artifacts.
_GDINO_BASE_SHORT_SIDE = 800
_GDINO_BASE_MAX_SIZE = 1333


def _resolve_gdino_scale(infer_scale):
    """Translate an infer_scale ratio into (short_side, max_size).

    Returns (None, None) when infer_scale is unset or equals 1.0 so that the
    underlying Model falls back to its bundled defaults (800 / 1333) and the
    code path is bit-identical to the pre-feature behavior.
    """
    if infer_scale is None:
        return None, None
    try:
        ratio = float(infer_scale)
    except (TypeError, ValueError):
        return None, None
    if not (ratio > 0):
        return None, None
    if abs(ratio - 1.0) < 1e-6:
        return None, None
    short = max(4, int(round(_GDINO_BASE_SHORT_SIDE * ratio / 4.0)) * 4)
    long_ = max(short, int(round(_GDINO_BASE_MAX_SIZE * ratio / 4.0)) * 4)
    return short, long_


class GDinoSAM3BaselineTracker(BaseOVTracker):
    """Baseline 4: Grounding DINO detection + SAM3 hard-restart segmentation."""

    def __init__(
        self,
        sam3_checkpoint,
        gdino_checkpoint,
        gdino_config=None,
        gdino_bert_path=None,
        relocation_interval=30,
        box_threshold=0.3,
        text_threshold=0.25,
        max_dets=None,
        nms_iou_thr=0.5,
        max_objects=20,
        new_det_thresh=0.85,
        max_trk_keep_alive=8,
        match_iou_thr=0.3,
        max_missing_segments=1,
        max_mask_area_ratio=0.5,
        sam3_tuning_cfg=None,
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
            sam3_checkpoint,
            max_objects=max_objects,
            new_det_thresh=new_det_thresh,
            max_trk_keep_alive=max_trk_keep_alive,
            match_iou_thr=match_iou_thr,
            max_missing_segments=max_missing_segments,
            max_mask_area_ratio=max_mask_area_ratio,
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
        self.relocation_interval = relocation_interval
        self.box_threshold = box_threshold
        # GroundingDINO's phrase parsing path still requires a numeric
        # text_threshold even though AeroTrack does not use the phrases later.
        self.text_threshold = 0.25 if text_threshold is None else text_threshold
        self.max_dets = max_dets
        self.nms_iou_thr = nms_iou_thr
        self.infer_scale = infer_scale
        self.gdino_model = None
        self._sam3_tuning_cfg = sam3_tuning_cfg or {}
        self.debug_gdino = self.verbose or (
            os.environ.get("AEROTRACK_DEBUG_GDINO", "0").lower() in ("1", "true", "yes")
        )

        self._next_tracker_obj_id = 0

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

    def detect_and_relocate(self, frame, text_prompt):
        """Run open-vocabulary detection on one frame with Grounding DINO."""
        detections, phrases = self.gdino_model.predict_with_caption(
            image=frame,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        if len(detections.xyxy) == 0:
            if self.debug_gdino:
                self._log(
                    f"[GDINO BL4] prompt='{text_prompt}' "
                    f"box_thr={self.box_threshold} text_thr={self.text_threshold} "
                    f"raw=0"
                )
            return [], []

        bboxes_np = detections.xyxy  # (N, 4) xyxy
        confs_np = detections.confidence  # (N,)
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
                f"[GDINO BL4] prompt='{text_prompt}' "
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

    def run_tracking(self, video_path, text_prompt, output_dir=None, prompt_frame_idx=0):
        self._text_prompt = text_prompt
        self._next_tracker_obj_id = 0
        return super().run_tracking(video_path, text_prompt, output_dir, prompt_frame_idx)

    def _add_boxes_as_prompt(self, session_id, frame_idx, img, bboxes, obj_ids=None, scores=None):
        """Inject Grounding DINO boxes into SAM3 via add_tracker_new_points."""
        if not bboxes:
            return

        H, W = img.shape[:2]
        inference_state = self.video_predictor._ALL_INFERENCE_STATES[session_id]["state"]
        model = self.video_predictor.model

        self._log(f"[BL4] Injecting {len(bboxes)} GDINO boxes on frame {frame_idx}")

        if "cached_frame_outputs" not in inference_state:
            inference_state["cached_frame_outputs"] = {}
        if frame_idx not in inference_state["cached_frame_outputs"]:
            inference_state["cached_frame_outputs"][frame_idx] = {}

        start_obj_id = self._next_tracker_obj_id
        score_iter = scores if scores is not None else [None] * len(bboxes)
        for (x1, y1, x2, y2), det_score in zip(bboxes, score_iter):
            x1 = max(0.0, float(x1))
            y1 = max(0.0, float(y1))
            x2 = min(float(x2), float(W))
            y2 = min(float(y2), float(H))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue

            box_norm = torch.tensor(
                [x1 / W, y1 / H, x2 / W, y2 / H], dtype=torch.float32
            )
            obj_id = self._next_tracker_obj_id
            self._next_tracker_obj_id += 1

            model.add_tracker_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box_norm,
                rel_coordinates=True,
            )
            self._set_local_obj_score(obj_id, det_score)

        text = getattr(self, "_text_prompt", None)
        if text:
            inference_state["text_prompt"] = text
            inference_state["input_batch"].find_text_batch[0] = text

        rank0_meta = inference_state["tracker_metadata"].get("rank0_metadata")
        if rank0_meta is not None:
            init_ka = getattr(model, "init_trk_keep_alive", 30)
            for oid in range(start_obj_id, self._next_tracker_obj_id):
                rank0_meta["obj_first_frame_idx"][oid] = frame_idx
                rank0_meta["trk_keep_alive"][oid] = init_ka

        inference_state["action_history"].clear()

        self._log(f"[BL4] {self._next_tracker_obj_id} total tracker objects registered.")
