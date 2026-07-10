"""Baseline 5: YOLO-World (detector) + SAM2 (segmenter), hard-restart.

Architecturally identical to BL3 (YOLO-World + SAM3) but uses SAM2 as the
segmentation backbone — a lighter alternative (857 MB vs 3.3 GB).
Detector logic is copied verbatim from YOLOWorldBaselineTracker (BL3).
"""

import os
import re
import tempfile

import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmdet.utils import get_test_pipeline_cfg
import mmdet.utils.setup_env  # noqa: F401
mmdet.utils.setup_env.register_all_modules()
from mmyolo.utils import register_all_modules as mmyolo_register_all_modules
mmyolo_register_all_modules(init_default_scope=False)
import mmyolo.models  # noqa: F401

import aerotrack_core.models.yolo_world.models.detectors.yolo_world  # noqa: F401
import aerotrack_core.models.yolo_world.models.detectors.yolo_world_image  # noqa: F401

from aerotrack_core.pipeline.sam2_tracker_base import BaseSAM2Tracker
from aerotrack_core.pipeline.yoloworld_baseline import (
    _patch_test_pipeline_scale,
    _resolve_yolo_img_size,
)


class YOLOWorldSAM2BaselineTracker(BaseSAM2Tracker):
    """BL5: YOLO-World + SAM2 hard-restart tracker."""

    _DEFAULT_CONFIG = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "../models/yolo_world/configs/pretrain/"
        "yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
    ))

    def __init__(
        self,
        sam2_checkpoint,
        yolo_checkpoint,
        yolo_config=None,
        relocation_interval=30,
        score_thr=0.25,
        max_dets=100,
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
        self.yolo_checkpoint = yolo_checkpoint
        self.yolo_config = yolo_config or self._DEFAULT_CONFIG
        self.score_thr = score_thr
        self.max_dets = max_dets
        self.nms_iou_thr = nms_iou_thr
        self.infer_scale = infer_scale
        self.yolo_model = None
        self.test_pipeline = None
        self.debug_yolo = self.verbose and (
            os.environ.get("AEROTRACK_DEBUG_YOLO", "1").lower() not in ("0", "false", "no")
        )

    # ------------------------------------------------------------------
    # NMS helpers (identical to BL3)
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
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
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
            try:
                from mmcv.ops import nms as mmcv_nms
                dets = torch.cat([bboxes, scores.unsqueeze(1)], dim=1)
                _, keep = mmcv_nms(dets, iou_thr)
                return keep
            except Exception:
                return self._nms_torch(bboxes, scores, iou_thr)

    def _build_test_pipeline(self, cfg):
        import copy

        if not hasattr(self, "_test_pipeline_cfg_template") or self._test_pipeline_cfg_template is None:
            self._test_pipeline_cfg_template = copy.deepcopy(get_test_pipeline_cfg(cfg=cfg))
        test_pipeline_cfg = copy.deepcopy(self._test_pipeline_cfg_template)
        test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        test_pipeline_cfg = [
            t for t in test_pipeline_cfg
            if t.get("type", "") not in ("LoadText", "RandomLoadText")
        ]
        new_img_size = _resolve_yolo_img_size(self.infer_scale)
        if new_img_size is not None:
            _patch_test_pipeline_scale(test_pipeline_cfg, new_img_size)
            self._log(
                f"[YOLO-World] infer_scale={self.infer_scale} -> "
                f"resize {new_img_size}x{new_img_size}"
            )
        return Compose(test_pipeline_cfg)

    def set_infer_scale(self, infer_scale):
        """Update YOLO preprocessing scale without reloading model weights."""
        try:
            infer_scale = float(infer_scale)
        except (TypeError, ValueError):
            return
        if abs(float(getattr(self, "infer_scale", 1.0)) - infer_scale) < 1e-6:
            return
        self.infer_scale = infer_scale
        if not hasattr(self, "_yolo_cfg") or self._yolo_cfg is None:
            return
        self.test_pipeline = self._build_test_pipeline(self._yolo_cfg)

    # ------------------------------------------------------------------
    # Config resolution (identical to BL3)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config(config_path):
        with open(config_path, "r") as f:
            content = f.read()
        if not re.search(r"../../third_party/mmyolo/", content):
            return config_path
        try:
            import mmyolo as _mmyolo
            mim_configs = os.path.join(os.path.dirname(_mmyolo.__file__), ".mim", "configs")
        except ImportError:
            mim_configs = None
        if not mim_configs or not os.path.isdir(mim_configs):
            raise FileNotFoundError("mmyolo .mim/configs not found. Run: mim install mmyolo==0.6.0")
        patched = re.sub(
            r"['\"]([^'\"]*third_party/mmyolo/configs/)([^'\"]+)['\"]",
            lambda m: f"'{os.path.join(mim_configs, m.group(2))}'",
            content,
        )
        patched = re.sub(
            r"custom_imports\s*=\s*dict\(([^)]*)\)",
            "custom_imports=dict(imports=['aerotrack_core.models.yolo_world'], allow_failed_imports=False)",
            patched, flags=re.MULTILINE,
        )
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=os.path.dirname(config_path)
        )
        tmp.write(patched)
        tmp.close()
        return tmp.name

    # ------------------------------------------------------------------
    # Detector loading (identical to BL3)
    # ------------------------------------------------------------------

    def _load_detector(self):
        self._log(f"Loading YOLO-World from {self.yolo_checkpoint}...")
        resolved = self._resolve_config(self.yolo_config)
        try:
            cfg = Config.fromfile(resolved)
        finally:
            if resolved != self.yolo_config and os.path.exists(resolved):
                os.unlink(resolved)
        cfg.work_dir = "./work_dirs"
        self._yolo_cfg = cfg
        self._test_pipeline_cfg_template = None
        model_cfg = cfg.get("model", cfg)
        if isinstance(model_cfg, dict) and "data_preprocessor" in model_cfg:
            dp_cfg = model_cfg["data_preprocessor"]
            if isinstance(dp_cfg, dict) and dp_cfg.get("type", "") in (
                "YOLOWDetDataPreprocessor", "YOLOWorldDetDataPreprocessor", "DetDataPreprocessor"
            ):
                model_cfg["data_preprocessor"] = dict(
                    type="YOLOv5DetDataPreprocessor",
                    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True,
                )
        from mmyolo.registry import MODELS as YOLO_MODELS, TASK_UTILS as YOLO_TASK_UTILS, TRANSFORMS as YOLO_TRANSFORMS
        from mmdet.registry import MODELS as DET_MODELS, TASK_UTILS as DET_TASK_UTILS, TRANSFORMS as DET_TRANSFORMS
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
                    for k in [k for k in d if k not in {"type", "loss_weight", "reduction", "eps", "mode"}]:
                        d.pop(k, None)
                for v in list(d.values()):
                    _clean_iou_loss_args(v)
            elif isinstance(d, list):
                for item in d:
                    _clean_iou_loss_args(item)

        _clean_iou_loss_args(model_cfg)
        self.yolo_model = DET_MODELS.build(model_cfg)
        load_checkpoint(self.yolo_model, self.yolo_checkpoint, map_location="cpu")
        self.yolo_model.to(self.device).eval()
        self.test_pipeline = self._build_test_pipeline(cfg)
        self._log("YOLO-World loaded.")

    # ------------------------------------------------------------------
    # Detection (identical to BL3)
    # ------------------------------------------------------------------

    def detect_and_relocate(self, frame, text_prompt):
        class_names = [t.strip() for t in text_prompt.split(",") if t.strip()]
        batched_texts = [class_names + [" "]]
        try:
            self.yolo_model.reparameterize(batched_texts)
        except Exception:
            self.yolo_model.reparameterize([batched_texts[0]])

        data_info = self.test_pipeline(dict(img=frame, img_id=0))

        if self.debug_yolo:
            try:
                inp = data_info["inputs"]
                inp_stats = inp.float() if not torch.is_floating_point(inp) else inp
                self._log(
                    f"[DEBUG YOLO-World] Input tensor: "
                    f"shape={tuple(inp.shape)} dtype={inp.dtype} "
                    f"min={float(inp_stats.min()):.4f} max={float(inp_stats.max()):.4f}"
                )
            except Exception:
                pass

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

            if self.debug_yolo and len(pred.scores) > 0:
                self._log(f"[DEBUG YOLO-World] Max score: {pred.scores.max().item():.4f}, "
                          f"above {self.score_thr}: {(pred.scores > self.score_thr).sum().item()}")

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

            return raw_bboxes.cpu().numpy().tolist(), pred.scores.cpu().numpy().tolist()

        except Exception as e:
            print(f"[ERROR YOLO-World BL5] Forward/PostProcess Exception: {e}")
            import traceback
            traceback.print_exc()
            return [], []
