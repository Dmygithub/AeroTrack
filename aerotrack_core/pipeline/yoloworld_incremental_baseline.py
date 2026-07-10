"""Baseline 9: Incremental YOLO-World + SAM3 tracker.

Uses YOLO-World as the external open-vocabulary detector at key frames.
Inherits the incremental single-session architecture from
``IncrementalOVTracker``.  YOLO-World loading and inference logic is
ported from ``YOLOWorldBaselineTracker`` (BL3).
"""

import os
import re
import tempfile

import cv2
import torch
import numpy as np
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmdet.utils import get_test_pipeline_cfg
from mmyolo.registry import MODELS as YOLO_MODELS
import mmdet.utils.setup_env  # noqa: F401
mmdet.utils.setup_env.register_all_modules()
from mmyolo.utils import register_all_modules as mmyolo_register_all_modules
mmyolo_register_all_modules(init_default_scope=False)
import mmyolo.models  # noqa: F401

import aerotrack_core.models.yolo_world.models.detectors.yolo_world  # noqa: F401
import aerotrack_core.models.yolo_world.models.detectors.yolo_world_image  # noqa: F401

from aerotrack_core.pipeline.incremental_tracker_base import IncrementalOVTracker


class YOLOWorldIncrementalTracker(IncrementalOVTracker):
    """BL9: YOLO-World (detector) + SAM3 (segmenter), incremental mode.

    Identical incremental architecture to BL7/BL8 but uses YOLO-World instead of
    SAM3's internal text detector at key frames.  YOLO-World typically provides
    faster and more stable open-vocabulary detection, especially for dense UAV
    scenes.
    """

    _DEFAULT_CONFIG = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "../models/yolo_world/configs/pretrain/"
        "yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
    ))

    def __init__(
        self,
        sam3_checkpoint,
        yolo_checkpoint,
        yolo_config=None,
        relocation_interval=30,
        match_iou_thr=0.3,
        score_thr=0.25,
        max_dets=100,
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
        self.yolo_checkpoint = yolo_checkpoint
        self.yolo_config = yolo_config or self._DEFAULT_CONFIG
        self.relocation_interval = relocation_interval
        self.score_thr = score_thr
        self.max_dets = max_dets
        self.nms_iou_thr = nms_iou_thr
        self.yolo_model = None
        self.test_pipeline = None
        self._sam3_tuning_cfg = sam3_tuning_cfg or {}

    # -- Model loading with optional SAM3 tuning ----------------------------

    def load_models(self):
        """Load SAM3 + YOLO-World and apply sam3_tuning overrides."""
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

    # -- BaseOVTracker abstract methods ------------------------------------

    def detect_and_relocate(self, frame, text_prompt):
        # Not used in incremental path; kept for interface compat.
        return None, None

    # -- YOLO-World loading (ported from BL3) ------------------------------

    @staticmethod
    def _resolve_config(config_path):
        """Rewrite _base_ relative paths in mmdet config to absolute paths."""
        with open(config_path, "r") as f:
            content = f.read()

        pattern = r"(../../third_party/mmyolo/)"
        if not re.search(pattern, content):
            return config_path

        try:
            import mmyolo as _mmyolo
            mim_configs = os.path.join(
                os.path.dirname(_mmyolo.__file__), ".mim", "configs"
            )
        except ImportError:
            mim_configs = None
        if not mim_configs or not os.path.isdir(mim_configs):
            raise FileNotFoundError(
                "mmyolo .mim/configs not found. Run: mim install mmyolo==0.6.0"
            )

        patched = re.sub(
            r"['\"]([^'\"]*third_party/mmyolo/configs/)([^'\"]+)['\"]",
            lambda m: f"'{os.path.join(mim_configs, m.group(2))}'",
            content,
        )
        patched = re.sub(
            r"custom_imports\s*=\s*dict\(([^)]*)\)",
            "custom_imports=dict(imports=['aerotrack_core.models.yolo_world'], allow_failed_imports=False)",
            patched,
            flags=re.MULTILINE,
        )

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=os.path.dirname(config_path),
        )
        tmp.write(patched)
        tmp.close()
        return tmp.name

    def _load_detector(self):
        """Load YOLO-World detector from config and checkpoint."""
        self._log(f"Loading YOLO-World from {self.yolo_checkpoint}...")
        resolved = self._resolve_config(self.yolo_config)
        try:
            cfg = Config.fromfile(resolved)
        finally:
            if resolved != self.yolo_config and os.path.exists(resolved):
                os.unlink(resolved)
        cfg.work_dir = "./work_dirs"

        model_cfg = cfg.get("model", cfg)
        if isinstance(model_cfg, dict) and "data_preprocessor" in model_cfg:
            dp_cfg = model_cfg["data_preprocessor"]
            if isinstance(dp_cfg, dict):
                dp_type = dp_cfg.get("type", "")
                if dp_type in (
                    "YOLOWDetDataPreprocessor",
                    "YOLOWorldDetDataPreprocessor",
                    "DetDataPreprocessor",
                ):
                    model_cfg["data_preprocessor"] = dict(
                        type="YOLOv5DetDataPreprocessor",
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        bgr_to_rgb=True,
                    )

        from mmyolo.registry import (
            MODELS as YOLO_MODELS,
            TASK_UTILS as YOLO_TASK_UTILS,
            TRANSFORMS as YOLO_TRANSFORMS,
        )
        from mmdet.registry import (
            MODELS as DET_MODELS,
            TASK_UTILS as DET_TASK_UTILS,
            TRANSFORMS as DET_TRANSFORMS,
        )

        for name, module in YOLO_MODELS.module_dict.items():
            if name not in DET_MODELS.module_dict:
                DET_MODELS.register_module(module=module, force=True)
        for name, module in YOLO_TASK_UTILS.module_dict.items():
            DET_TASK_UTILS.register_module(name=name, module=module, force=True)
        for name, module in YOLO_TRANSFORMS.module_dict.items():
            if name not in DET_TRANSFORMS.module_dict:
                DET_TRANSFORMS.register_module(module=module, force=True)

        def _clean_iou_loss_args(d):
            if isinstance(d, dict):
                if d.get("type") == "IoULoss" or "loss" in d.get("type", "").lower():
                    safe_keys = {"type", "loss_weight", "reduction", "eps", "mode"}
                    for k in [k for k in d if k not in safe_keys]:
                        d.pop(k, None)
                for v in d.values():
                    _clean_iou_loss_args(v)
            elif isinstance(d, list):
                for item in d:
                    _clean_iou_loss_args(item)

        _clean_iou_loss_args(model_cfg)

        self.yolo_model = DET_MODELS.build(model_cfg)
        load_checkpoint(self.yolo_model, self.yolo_checkpoint, map_location="cpu")
        self.yolo_model.to(self.device)
        self.yolo_model.eval()
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        test_pipeline_cfg = [
            t for t in test_pipeline_cfg
            if t.get("type", "") not in ("LoadText", "RandomLoadText")
        ]
        self.test_pipeline = Compose(test_pipeline_cfg)
        self._log("YOLO-World loaded.")

    # -- NMS helpers (ported from BL3) -------------------------------------

    @staticmethod
    def _nms_torch(bboxes, scores, iou_thr):
        """Pure PyTorch NMS fallback (xyxy boxes)."""
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
        if not keep:
            return torch.empty((0,), dtype=torch.long, device=bboxes.device)
        return torch.stack(keep)

    def _apply_nms(self, bboxes, scores, iou_thr=0.5):
        """Apply NMS with best-available backend."""
        if bboxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=bboxes.device)
        try:
            from torchvision.ops import nms
            return nms(bboxes, scores, iou_thr)
        except Exception:
            return self._nms_torch(bboxes, scores, iou_thr)

    # -- IncrementalOVTracker abstract method ------------------------------

    def detect_on_keyframe(self, frame, frame_idx, text_prompt, inference_state):
        """Run YOLO-World detection on one frame.

        Supports multi-category prompts via comma separation (e.g. "car, van").
        Applies score thresholding, top-k capping, and NMS.
        """
        class_names = [t.strip() for t in text_prompt.split(",") if t.strip()]
        batched_texts = [class_names + [" "]]

        try:
            self.yolo_model.reparameterize(batched_texts)
        except Exception:
            self.yolo_model.reparameterize([batched_texts[0]])

        data_info = dict(img=frame, img_id=0)
        data_info = self.test_pipeline(data_info)

        data_samples = data_info["data_samples"]
        if hasattr(data_samples, "texts"):
            delattr(data_samples, "texts")
        if "texts" in data_samples:
            data_samples.pop("texts", None)
        if "pad_param" in data_samples:
            pad = data_samples.get("pad_param")
            if pad is not None and len(pad) < 4:
                data_samples.set_metainfo({"pad_param": None})

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_samples],
        )

        try:
            with torch.no_grad():
                processed = self.yolo_model.data_preprocessor(data_batch, False)
                output = self.yolo_model.predict(
                    processed["inputs"], processed["data_samples"], rescale=True
                )[0]

            pred = output.pred_instances

            keep = pred.scores > self.score_thr
            pred = pred[keep]
            if len(pred) == 0:
                return [], []

            # 1. First apply NMS
            keep_nms = self._apply_nms(pred.bboxes, pred.scores, iou_thr=self.nms_iou_thr)
            pred = pred[keep_nms]

            # 2. Then apply max_dets (Top-K)
            max_dets = int(self.max_dets) if self.max_dets is not None else len(pred)
            if max_dets > 0 and len(pred) > max_dets:
                _, topk_idx = pred.scores.topk(max_dets)
                pred = pred[topk_idx]

            ori_h, ori_w = frame.shape[:2]
            raw_bboxes = pred.bboxes.clone().float()
            raw_bboxes = raw_bboxes.clamp(min=0)
            raw_bboxes[:, 0::2] = raw_bboxes[:, 0::2].clamp(max=ori_w)
            raw_bboxes[:, 1::2] = raw_bboxes[:, 1::2].clamp(max=ori_h)
            raw_bboxes = raw_bboxes.cpu().numpy()

            bboxes = raw_bboxes.tolist()
            confs = pred.scores.cpu().numpy().tolist()
            return bboxes, confs
        except Exception as e:
            print(f"[ERROR BL9 YOLO-World] {e}")
            import traceback
            traceback.print_exc()
            return [], []
