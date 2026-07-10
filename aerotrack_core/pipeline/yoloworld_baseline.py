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

from aerotrack_core.pipeline.ov_tracker_base import BaseOVTracker


# YOLOv5 backbone uses stride=32; resolution must be a multiple of 32.
_YOLO_BASE_IMG_SIZE = 640


def _resolve_yolo_img_size(infer_scale, base_size=_YOLO_BASE_IMG_SIZE):
    """Translate an infer_scale ratio into a stride-aligned img_size.

    Returns None when infer_scale is unset or equals 1.0 so that the underlying
    config-provided img_scale is left untouched (bit-identical pre-feature
    behavior).
    """
    if infer_scale is None:
        return None
    try:
        ratio = float(infer_scale)
    except (TypeError, ValueError):
        return None
    if not (ratio > 0):
        return None
    if abs(ratio - 1.0) < 1e-6:
        return None
    return max(32, int(round(base_size * ratio / 32.0)) * 32)


def _patch_test_pipeline_scale(test_pipeline_cfg, new_scale):
    """Override every transform's ``scale`` field to ``(new_scale, new_scale)``.

    YOLO-World test pipelines use ``YOLOv5KeepRatioResize`` and ``LetterResize``
    transforms keyed by ``scale``. mmengine ConfigDict items are mutable, so we
    update them in place.
    """
    target = (new_scale, new_scale)
    for entry in test_pipeline_cfg:
        if isinstance(entry, dict) and "scale" in entry:
            entry["scale"] = target


class YOLOWorldBaselineTracker(BaseOVTracker):
    """Baseline 3: YOLO-World (Detector) + SAM3 (Segmenter/Tracker).

    Open-vocabulary two-stage framework:
        1. Detection: YOLO-World detects objects on each key frame using a
           text prompt and injects bounding boxes into SAM3.
        2. Segmentation & Tracking: SAM3 segments and propagates masks between
           key frames via ``propagation_full``.
        3. Periodic Re-detection: every ``relocation_interval`` frames a new
           key frame fires YOLO-World to discover new / re-entered targets.

    Inherits ``BaseOVTracker.run_tracking`` for segment-based pipeline,
    global-ID remapping, and memory pruning.

    Args:
        sam3_checkpoint: Path to SAM3 model weights.
        yolo_checkpoint: Path to YOLO-World checkpoint (.pth).
        yolo_config: Path to mmdet config .py; defaults to YOLO-World v2-L.
        relocation_interval: Frames between periodic key frames.
        score_thr: YOLO-World detection confidence threshold.
        max_dets: Max detections injected per key frame.
        nms_iou_thr: NMS IoU threshold for YOLO-World post-processing.
        max_objects: Cap on SAM3 simultaneously tracked instances.
        new_det_thresh: SAM3 internal new-track confidence threshold.
        max_trk_keep_alive: Frames a lost SAM3 track survives before removal.
        sam3_tuning_cfg: Optional dict of SAM3 model parameter overrides.
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
        score_thr=0.25,
        max_dets=100,
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
        self.yolo_checkpoint = yolo_checkpoint
        self.yolo_config = yolo_config or self._DEFAULT_CONFIG
        self.relocation_interval = relocation_interval
        self.score_thr = score_thr
        self.max_dets = max_dets
        self.nms_iou_thr = nms_iou_thr
        self.infer_scale = infer_scale
        self.yolo_model = None
        self.test_pipeline = None
        self._sam3_tuning_cfg = sam3_tuning_cfg or {}

        self.debug_yolo = self.verbose and (
            os.environ.get("AEROTRACK_DEBUG_YOLO", "1").lower() not in ("0", "false", "no")
        )

        self._next_tracker_obj_id = 0

    # ------------------------------------------------------------------
    # NMS helpers (used by detect_and_relocate)
    # ------------------------------------------------------------------

    @staticmethod
    def _nms_torch(bboxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
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
        if len(keep) == 0:
            return torch.empty((0,), dtype=torch.long, device=bboxes.device)
        return torch.stack(keep)

    def _apply_nms(self, bboxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
        """Apply NMS with best-available backend; returns keep indices."""
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
    # Config resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config(config_path):
        """Rewrite _base_ relative paths in a mmdet config to absolute paths.

        The pretrain configs use ``../../third_party/mmyolo/configs/...`` which
        only exists in the original YOLO-World repo tree.  At inference time the
        base configs live inside the installed mmyolo package under ``.mim/``.
        This helper patches the ``_base_`` line in-place by writing a temp file
        so that mmengine can resolve the inheritance chain.

        Args:
            config_path: Absolute path to the mmdet config `.py` file.

        Returns:
            Path to a (possibly patched) config file safe to pass to
            ``Config.fromfile``.
        """
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
            flags=re.MULTILINE
        )

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=os.path.dirname(config_path)
        )
        tmp.write(patched)
        tmp.close()
        return tmp.name

    # ------------------------------------------------------------------
    # Detector loading
    # ------------------------------------------------------------------

    def _load_detector(self):
        """Load YOLO-World detector from config and checkpoint without datasets."""
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
            if isinstance(dp_cfg, dict):
                dp_type = dp_cfg.get("type", "")
                if dp_type in ("YOLOWDetDataPreprocessor", "YOLOWorldDetDataPreprocessor", "DetDataPreprocessor"):
                    model_cfg["data_preprocessor"] = dict(
                        type="YOLOv5DetDataPreprocessor",
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        bgr_to_rgb=True,
                    )

        from mmengine.registry import Registry
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
                if d.get('type') == 'IoULoss' or 'loss' in d.get('type', '').lower():
                    safe_keys = {'type', 'loss_weight', 'reduction', 'eps', 'mode'}
                    keys_to_remove = [k for k in d.keys() if k not in safe_keys]
                    for k in keys_to_remove:
                        d.pop(k, None)
                for k, v in list(d.items()):
                    _clean_iou_loss_args(v)
            elif isinstance(d, list):
                for item in d:
                    _clean_iou_loss_args(item)

        _clean_iou_loss_args(model_cfg)

        self.yolo_model = DET_MODELS.build(model_cfg)

        load_checkpoint(self.yolo_model, self.yolo_checkpoint, map_location="cpu")
        self.yolo_model.to(self.device)
        self.yolo_model.eval()
        self.test_pipeline = self._build_test_pipeline(cfg)
        self._log("YOLO-World loaded.")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_and_relocate(self, frame, text_prompt):
        """Run YOLO-World on a single frame with open-vocabulary text prompt.

        Supports multi-category prompts via comma separation, e.g. "car, van".
        Applies score thresholding, top-k capping, and NMS.

        Args:
            frame: BGR numpy image (H, W, 3).
            text_prompt: Comma-separated category string.

        Returns:
            bboxes: List of [x1, y1, x2, y2] pixel-coordinate boxes.
            confs: List of float confidence scores.
        """
        class_names = [t.strip() for t in text_prompt.split(",") if t.strip()]
        batched_texts = [class_names + [" "]]

        try:
            self.yolo_model.reparameterize(batched_texts)
        except Exception:
            flat_texts = batched_texts[0]
            self.yolo_model.reparameterize([flat_texts])

        data_info = dict(img=frame, img_id=0)
        data_info = self.test_pipeline(data_info)
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
        if 'texts' in data_samples:
            data_samples.pop("texts", None)
        if 'pad_param' in data_samples:
            pad = data_samples.get('pad_param')
            if pad is not None and len(pad) < 4:
                data_samples.set_metainfo({'pad_param': None})

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

            # rescale=True: head already maps bboxes to original image space
            # (pad_param subtracted + scale_factor applied in predict_by_feat)
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
            print(f"[ERROR YOLO-World Baseline] Forward/PostProcess Exception: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    # ------------------------------------------------------------------
    # Model loading with optional SAM3 tuning
    # ------------------------------------------------------------------

    def load_models(self):
        """Load SAM3 + YOLO-World and apply sam3_tuning overrides if provided."""
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
    # Capture text_prompt for VG detection matching
    # ------------------------------------------------------------------

    def run_tracking(self, video_path, text_prompt, output_dir=None, prompt_frame_idx=0):
        """Capture text_prompt for _add_boxes_as_prompt."""
        self._text_prompt = text_prompt
        self._next_tracker_obj_id = 0
        return super().run_tracking(video_path, text_prompt, output_dir, prompt_frame_idx)

    # ------------------------------------------------------------------
    # Box injection via SAM3 tracker path (overrides base class)
    # ------------------------------------------------------------------

    def _add_boxes_as_prompt(self, session_id, frame_idx, img, bboxes, obj_ids=None, scores=None):
        """Inject YOLO boxes as individual SAM3 tracker objects.

        Each box is registered via ``add_tracker_new_points`` so that SAM3's
        tracker creates a dedicated per-object state.  The text prompt is set
        on the inference state so the VG detector provides detection-to-track
        matching during propagation, keeping tracker objects alive through
        SAM3's hotstart mechanism.

        Args:
            session_id: Active SAM3 session ID.
            frame_idx: Frame index for the prompt.
            img: BGR image array (H, W, 3) for coordinate normalization.
            bboxes: List of [x1, y1, x2, y2] pixel-coordinate boxes.
            obj_ids: Unused; kept for API compatibility.
            scores: Optional detector confidences aligned with bboxes.
        """
        if not bboxes:
            return

        H, W = img.shape[:2]
        inference_state = self.video_predictor._ALL_INFERENCE_STATES[session_id]["state"]
        model = self.video_predictor.model

        self._log(f"[BL3] Injecting {len(bboxes)} YOLO boxes on frame {frame_idx}")
        if bboxes:
            b0 = bboxes[0]
            self._log(f"[BL3] box[0] type={type(b0).__name__}, val={b0}, "
                      f"w={b0[2]-b0[0]:.1f}, h={b0[3]-b0[1]:.1f}")

        # Ensure cached_frame_outputs exists for this frame so that
        # _build_tracker_output inside add_tracker_new_points doesn't assert.
        if "cached_frame_outputs" not in inference_state:
            inference_state["cached_frame_outputs"] = {}
        if frame_idx not in inference_state["cached_frame_outputs"]:
            inference_state["cached_frame_outputs"][frame_idx] = {}

        start_obj_id = self._next_tracker_obj_id
        score_iter = list(scores) if scores is not None else []
        if len(score_iter) < len(bboxes):
            score_iter.extend([None] * (len(bboxes) - len(score_iter)))
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

        # Enable VG text detection for detection-to-track matching.
        text = getattr(self, "_text_prompt", None)
        if text:
            inference_state["text_prompt"] = text
            inference_state["input_batch"].find_text_batch[0] = text

        # Register injected objects in rank0_metadata.
        rank0_meta = inference_state["tracker_metadata"].get("rank0_metadata")
        if rank0_meta is not None:
            init_ka = getattr(model, "init_trk_keep_alive", 30)
            for oid in range(start_obj_id, self._next_tracker_obj_id):
                rank0_meta["obj_first_frame_idx"][oid] = frame_idx
                rank0_meta["trk_keep_alive"][oid] = init_ka

        # Force propagation_full on the next propagate_in_video call.
        inference_state["action_history"].clear()

        self._log(f"[BL3] {self._next_tracker_obj_id} total tracker objects registered.")
