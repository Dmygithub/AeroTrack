"""SAM2-based tracker base class for BL5 / BL6.

Mirrors the hard-restart segment architecture of BaseOVTracker (BL2/BL3/BL4)
but drives SAM2VideoPredictor instead of Sam3VideoPredictor.

Key improvements over a naive box-prompt approach (following Grounded-SAM-2):
  - Detection boxes are first refined by SAM2ImagePredictor into pixel-accurate
    masks on each key frame (mask prompt vs. box prompt gives a better init).
  - Within-detection-set mask-IoU NMS removes duplicate objects (e.g. the same
    truck detected as cab + cargo box, or a person split into 5-6 boxes).
  - Refined masks are injected via add_new_mask (mask prompt) rather than
    add_new_points_or_box (box prompt) so the video propagation starts from a
    cleaner initialization.

SAM2 API (differs from SAM3):
  - Session  : init_state(video_path) -> inference_state dict  (no session_id)
  - Img pred : SAM2ImagePredictor wraps the same model object  (zero extra VRAM)
  - Inject   : add_new_mask(state, frame_idx, obj_id, mask_2d)
  - Propagate: propagate_in_video(state) -> generator of (frame_idx, obj_ids, logits)
  - Reset    : reset_state(state)
"""

import os
import gc
import time
import tempfile
import glob as _glob
from abc import ABC, abstractmethod

import cv2
import torch
import numpy as np

os.environ["LOG_LEVEL"] = "ERROR"
import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

from aerotrack_core.pipeline.id_association import build_video_id_association

# Hydra must be initialised before build_sam2_video_predictor is called.
_HYDRA_INITIALISED = False

SAM2_CONFIG = "sam2.1_hiera_l"   # config name (no .yaml suffix)


def _ensure_hydra(config_dir: str):
    global _HYDRA_INITIALISED
    if _HYDRA_INITIALISED:
        return
    from hydra import initialize_config_dir
    initialize_config_dir(config_dir=config_dir, job_name="aerotrack_sam2", version_base="1.2")
    _HYDRA_INITIALISED = True


class BaseSAM2Tracker(ABC):
    """Base class for SAM2-backed hard-restart trackers (BL5, BL6).

    Args:
        sam2_checkpoint: Path to sam2.1 model weights.
        relocation_interval: Frames between consecutive key frames.
        max_objects: Max objects injected per key frame.
        mask_nms_iou_thr: Mask-IoU threshold for within-detection NMS.
            Boxes whose resulting masks overlap above this threshold are merged
            (keep highest-score). Default 0.5 matches Grounded-SAM-2 practice.
        verbose: Print debug messages if True.
        show_progress: Show tqdm progress bar if True.
    """

    def __init__(
        self,
        sam2_checkpoint,
        relocation_interval=50,
        max_objects=20,
        match_iou_thr=0.3,
        max_mask_area_ratio=0.5,
        mask_nms_iou_thr=0.5,
        use_track_lifecycle=True,
        lost_track_ttl_segments=2,
        lifecycle_match_score_thr=0.45,
        lifecycle_center_gate=2.5,
        lifecycle_area_ratio_min=0.35,
        lifecycle_area_ratio_max=2.8,
        verbose=False,
        show_progress=True,
    ):
        self.sam2_checkpoint = sam2_checkpoint
        self.relocation_interval = relocation_interval
        self.max_objects = max_objects
        self.match_iou_thr = match_iou_thr
        self.max_mask_area_ratio = max_mask_area_ratio
        self.mask_nms_iou_thr = mask_nms_iou_thr
        self.use_track_lifecycle = use_track_lifecycle
        self.lost_track_ttl_segments = lost_track_ttl_segments
        self.lifecycle_match_score_thr = lifecycle_match_score_thr
        self.lifecycle_center_gate = lifecycle_center_gate
        self.lifecycle_area_ratio_min = lifecycle_area_ratio_min
        self.lifecycle_area_ratio_max = lifecycle_area_ratio_max
        self.verbose = verbose
        self.show_progress = show_progress

        self.video_predictor = None
        self.image_predictor = None   # SAM2ImagePredictor wrapping video_predictor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _cleanup_gpu(self):
        gc.collect()
        if torch.cuda.is_available() and not os.environ.get("AEROTRACK_MEMORY_RESERVE"):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_sam2(self):
        """Load SAM2VideoPredictor and wrap it with SAM2ImagePredictor.

        The image predictor shares the same underlying SAM2Base model object as
        the video predictor, so no extra weights are loaded (zero VRAM overhead
        beyond the video predictor itself).
        """
        from aerotrack_core.models.sam2.build_sam import build_sam2_video_predictor
        from aerotrack_core.models.sam2.sam2_image_predictor import SAM2ImagePredictor

        config_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "models", "sam2", "configs", "sam2.1"
        ))
        _ensure_hydra(config_dir)

        self._log(f"Loading SAM2 from {self.sam2_checkpoint} ...")
        self.video_predictor = build_sam2_video_predictor(
            config_file=SAM2_CONFIG,
            ckpt_path=self.sam2_checkpoint,
            device=self.device,
        )
        # Wrap with image predictor — shares model weights, costs no extra VRAM.
        self.image_predictor = SAM2ImagePredictor(self.video_predictor)
        self._log("SAM2 (video + image predictor) loaded.")

    def load_models(self):
        """Load SAM2 models and the subclass-specific detector."""
        self._cleanup_gpu()
        self._load_sam2()
        self._load_detector()
        self._log("All models loaded.")

    @abstractmethod
    def _load_detector(self):
        pass

    @abstractmethod
    def detect_and_relocate(self, frame, text_prompt):
        """Return (bboxes [[x1,y1,x2,y2],...], confidences [float,...])."""
        pass

    # ------------------------------------------------------------------
    # SAM2 session helpers
    # ------------------------------------------------------------------

    def _init_session(self, video_path):
        """Create a new SAM2 inference state for the given frame directory."""
        return self.video_predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )

    def _reset_session(self, inference_state):
        self.video_predictor.reset_state(inference_state)

    # ------------------------------------------------------------------
    # Frame I/O
    # ------------------------------------------------------------------

    def _read_frame(self, frames_list, video_path, frame_idx):
        """Read a single BGR frame from a frame list or video file."""
        if frames_list is not None:
            return cv2.imread(frames_list[frame_idx])
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, img = cap.read()
        cap.release()
        return img

    # ------------------------------------------------------------------
    # Image Predictor helpers: boxes -> masks -> NMS -> inject
    # ------------------------------------------------------------------

    def _boxes_to_masks(self, frame_bgr, bboxes):
        """Run SAM2ImagePredictor to convert detector boxes to pixel masks.

        All boxes are processed in a single forward pass using the batched
        box-prompt interface, matching the Grounded-SAM-2 reference design.

        Args:
            frame_bgr: BGR numpy image (H, W, 3) from cv2.
            bboxes: List of [x1, y1, x2, y2] pixel-coordinate boxes.

        Returns:
            masks: numpy bool array (N, H, W).
            scores: numpy float array (N,) — mask quality scores from SAM2.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.image_predictor.set_image(frame_rgb)

        boxes_np = np.array(bboxes, dtype=np.float32)  # (N, 4) xyxy
        with torch.autocast("cuda", dtype=torch.bfloat16):
            masks_out, scores_out, _ = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_np,
                multimask_output=True,
            )
        # multimask_output=True: SAM2 returns 3 candidates per box, pick best.
        # SAM2 predict() squeeze(0) behaviour, actual shapes:
        #   N=1 : masks (3, H, W),   scores (3,)
        #   N>1 : masks (N, 3, H, W), scores (N, 3)
        if masks_out.ndim == 4:          # N>1: pick best of 3 per box
            best = scores_out.argmax(axis=1)          # (N,)
            masks_out  = masks_out[np.arange(len(best)), best]   # (N, H, W)
            scores_out = scores_out[np.arange(len(best)), best]  # (N,)
        else:                            # N=1: (3, H, W) → (1, H, W)
            best = int(scores_out.argmax())
            masks_out  = masks_out[best:best+1]
            scores_out = scores_out[best:best+1]
        return masks_out.astype(bool), scores_out.astype(float)

    @staticmethod
    def _mask_nms(masks, scores, iou_thr=0.5):
        """Greedy mask-IoU NMS within a single detection set.

        Processes masks in descending score order. A candidate mask is
        suppressed if its IoU with any already-kept mask exceeds iou_thr.
        This removes duplicate detections (e.g. truck cab + cargo box, or
        multiple overlapping person boxes on one scooter).

        Args:
            masks: bool numpy array (N, H, W).
            scores: float numpy array (N,).
            iou_thr: Suppress if mask-IoU > this value.

        Returns:
            keep: list of int indices into masks/scores to retain.
        """
        if len(masks) == 0:
            return []
        order = np.argsort(scores)[::-1]
        keep = []
        for i in order:
            dominated = False
            for j in keep:
                inter = int((masks[i] & masks[j]).sum())
                union = int((masks[i] | masks[j]).sum())
                if union > 0 and inter / union > iou_thr:
                    dominated = True
                    break
            if not dominated:
                keep.append(int(i))
        return keep

    def _inject_masks(self, inference_state, frame_idx, frame_bgr, bboxes, det_scores=None):
        """Refine detected boxes with SAM2ImagePredictor and inject as mask prompts.

        Pipeline (following Grounded-SAM-2 reference):
          1. SAM2ImagePredictor converts each detected box to a pixel mask.
          2. Mask-IoU NMS removes duplicates within the current detection set.
          3. Surviving masks are registered via add_new_mask (mask prompt),
             giving the video predictor a high-quality initialization.

        Args:
            inference_state: Active SAM2 inference state (after reset_state).
            frame_idx: Index of the key frame in the video.
            frame_bgr: BGR numpy image of the key frame (for Image Predictor).
            bboxes: List of [x1, y1, x2, y2] boxes (already NMS-filtered by
                the detector, capped to max_objects).
            det_scores: Optional detector confidences aligned with bboxes.

        Returns:
            obj_ids: List of integer object IDs registered in SAM2.
            obj_scores: Dict mapping local obj_id to detector confidence.
        """
        if not bboxes:
            return [], {}

        bboxes = bboxes[: self.max_objects]
        if det_scores is not None:
            det_scores = list(det_scores)[: self.max_objects]
        else:
            det_scores = [None] * len(bboxes)

        # Step 1: boxes → pixel masks via SAM2 Image Predictor
        try:
            masks, scores = self._boxes_to_masks(frame_bgr, bboxes)
        except Exception as e:
            self._log(f"[SAM2 ImagePredictor] Failed: {e}; falling back to box prompts.")
            return self._inject_boxes_fallback(
                inference_state, frame_idx, bboxes, det_scores=det_scores
            )
        finally:
            self.image_predictor.reset_predictor()

        # Step 2: mask-IoU NMS to remove intra-set duplicates
        keep = self._mask_nms(masks, scores, iou_thr=self.mask_nms_iou_thr)
        self._log(
            f"[BL5/6] key frame {frame_idx}: {len(bboxes)} boxes → "
            f"{len(masks)} masks → {len(keep)} after mask-NMS"
        )

        # Step 3: inject surviving masks via add_new_mask
        obj_ids = []
        obj_scores = {}
        for i, ki in enumerate(keep):
            obj_id = i + 1
            mask_2d = masks[ki]   # (H, W) bool
            self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask_2d,
            )
            obj_ids.append(obj_id)
            det_score = det_scores[ki] if ki < len(det_scores) else None
            if det_score is not None:
                try:
                    det_score = float(det_score)
                    if np.isfinite(det_score):
                        obj_scores[obj_id] = float(np.clip(det_score, 0.0, 1.0))
                except (TypeError, ValueError):
                    pass

        return obj_ids, obj_scores

    def _inject_boxes_fallback(self, inference_state, frame_idx, bboxes, det_scores=None):
        """Fallback: inject raw boxes if Image Predictor fails.

        Args:
            inference_state: Active SAM2 inference state.
            frame_idx: Key frame index.
            bboxes: List of [x1, y1, x2, y2] boxes.
            det_scores: Optional detector confidences aligned with bboxes.

        Returns:
            obj_ids: List of registered object IDs.
            obj_scores: Dict mapping local obj_id to detector confidence.
        """
        obj_ids = []
        obj_scores = {}
        if det_scores is not None:
            det_scores = list(det_scores)
        else:
            det_scores = [None] * len(bboxes)
        for i, (x1, y1, x2, y2) in enumerate(bboxes[: self.max_objects]):
            obj_id = i + 1
            box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
            )
            obj_ids.append(obj_id)
            det_score = det_scores[i] if i < len(det_scores) else None
            if det_score is not None:
                try:
                    det_score = float(det_score)
                    if np.isfinite(det_score):
                        obj_scores[obj_id] = float(np.clip(det_score, 0.0, 1.0))
                except (TypeError, ValueError):
                    pass
        return obj_ids, obj_scores

    # ------------------------------------------------------------------
    # Memory pruning
    # ------------------------------------------------------------------

    @staticmethod
    def _sam2_prune_memory(inference_state: dict, current_frame_idx: int, window: int = 7) -> None:
        """Trim SAM2 non-conditioning outputs older than `window` frames.

        SAM2 accumulates per-object non-cond outputs in both
        ``output_dict_per_obj`` and ``temp_output_dict_per_obj`` indefinitely.
        On long videos this causes unbounded CPU RAM growth (~several MB/frame).
        Deletes entries whose frame index is older than
        ``current_frame_idx - window``, matching SAM2's own maskmem window.
        """
        cutoff = current_frame_idx - window
        if cutoff <= 0:
            return
        for dict_key in ("output_dict_per_obj", "temp_output_dict_per_obj"):
            per_obj = inference_state.get(dict_key, {})
            for obj_dict in per_obj.values():
                non_cond = obj_dict.get("non_cond_frame_outputs", {})
                stale = [k for k in list(non_cond.keys()) if isinstance(k, int) and k < cutoff]
                for k in stale:
                    del non_cond[k]

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_tracking(self, video_path, text_prompt, output_dir=None, prompt_frame_idx=0):
        """Run hard-restart segmentation tracking with SAM2.

        For each segment [kf, kf+relocation_interval):
          1. Detector fires on key frame kf.
          2. SAM2ImagePredictor refines boxes to masks; mask-NMS deduplicates.
          3. Masks injected via add_new_mask into a fresh SAM2 session.
          4. propagate_in_video tracks objects through the segment.
          5. Lifecycle-aware ID association keeps video-level IDs stable.

        Args:
            video_path: Path to video file or directory of JPEG frames.
            text_prompt: Open-vocabulary text category for the detector.
            output_dir: Unused; kept for API parity with BaseOVTracker.
            prompt_frame_idx: Frame index to start tracking from.

        Returns:
            (temp_cache_dir, total_frames, fps, (W, H))
        """
        self._cleanup_gpu()

        # --- Resolve video source ---
        if os.path.isdir(video_path):
            frames_list = []
            for ext in ("jpg", "jpeg", "png"):
                frames_list.extend(_glob.glob(os.path.join(video_path, f"*.{ext}")))
            # Deduplicate by basename then sort — guards against a directory
            # containing both .jpg and .jpeg variants of the same frame index.
            seen = set()
            deduped = []
            for p in frames_list:
                bn = os.path.basename(p)
                if bn not in seen:
                    seen.add(bn)
                    deduped.append(p)
            frames_list = sorted(deduped)
            if not frames_list:
                raise RuntimeError(f"No image frames found in directory: {video_path}")
            first_frame = cv2.imread(frames_list[0])
            total_frames = len(frames_list)
            fps = 30.0
        else:
            frames_list = None
            cap = cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            if not ret or first_frame is None:
                raise RuntimeError(f"Cannot read first frame: {video_path}")

        H, W = first_frame.shape[:2]
        reloc_interval = int(getattr(self, "relocation_interval", 0) or 0)
        if reloc_interval > 0:
            key_frames = list(range(prompt_frame_idx, total_frames, reloc_interval))
        else:
            key_frames = [prompt_frame_idx]

        temp_cache_dir = tempfile.mkdtemp(prefix="tracker_cache_")
        t0 = time.time()

        id_assoc = build_video_id_association(self, frame_area=H * W)

        # SAM2 needs a frame directory; dump video to temp dir if needed.
        frame_dir = None
        _owns_frame_dir = False
        if frames_list is None:
            frame_dir = tempfile.mkdtemp(prefix="sam2_frames_")
            _owns_frame_dir = True
            cap = cv2.VideoCapture(video_path)
            fidx = 0
            while True:
                ret, frm = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(frame_dir, f"{fidx:06d}.jpg"), frm)
                fidx += 1
            cap.release()
            # Build a frames_list from the dumped jpegs so _read_frame uses
            # the already-decoded files instead of re-opening the video.
            frames_list_for_read = sorted(
                _glob.glob(os.path.join(frame_dir, "*.jpg"))
            )
        else:
            frame_dir = os.path.dirname(frames_list[0])
            frames_list_for_read = frames_list

        inference_state = self._init_session(frame_dir)
        # Expose for OOM cleanup in inference_engine (SAM2 has no session-id API).
        self._sam2_inference_state = inference_state

        # Progress bar — mirrors BL2/3/4 behaviour.
        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm as _tqdm
                pbar = _tqdm(
                    total=max(0, total_frames - prompt_frame_idx),
                    desc="Tracking",
                    unit="frame",
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:
                pbar = None

        count = 0
        try:
            for seg_idx, kf in enumerate(key_frames):
                seg_end = key_frames[seg_idx + 1] if seg_idx + 1 < len(key_frames) else total_frames

                # 1. Detect on key frame
                kf_img = self._read_frame(frames_list_for_read, None, kf)
                bboxes, det_confs = self.detect_and_relocate(kf_img, text_prompt)

                # 2. Reset SAM2 state for this segment
                self._reset_session(inference_state)

                # Detection failed: skip segment entirely (no empty npz written).
                # Consistent with BL3/BL4 behaviour so coverage scores are comparable.
                if not bboxes:
                    self._log(f"[BL5/6] Key frame {kf}: no detections, skipping segment")
                    id_assoc.end_segment(seg_idx, {})
                    if pbar is not None:
                        pbar.update(seg_end - kf)
                    continue

                # 3. Refine boxes → masks (Image Predictor + mask NMS) and inject
                local_obj_scores = {}
                _obj_ids, local_obj_scores = self._inject_masks(
                    inference_state, kf, kf_img, bboxes, det_scores=det_confs
                )

                # 4. Propagate through segment
                seg_last_bboxes_local = {}

                with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                    for frame_idx, obj_ids_out, mask_logits in self.video_predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=kf,
                        max_frame_num_to_track=seg_end - kf,
                    ):
                        # mask_logits shape: (N, 1, H, W) — squeeze the mask dim
                        if mask_logits.ndim == 4:
                            mask_logits = mask_logits[:, 0]   # (N, H, W)
                        masks_hw = (mask_logits > 0.0).cpu().numpy()  # (N, H, W) bool

                        # Resize to original video resolution if needed
                        masks_full = []
                        for m in masks_hw:
                            if m.shape != (H, W):
                                m = cv2.resize(
                                    m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                                ).astype(bool)
                            masks_full.append(m)

                        # Register local SAM2 obj_ids as video-level global ids.
                        ref = id_assoc.previous_bboxes() if frame_idx == kf else {}
                        id_assoc.assign_frame(
                            seg_idx,
                            obj_ids_out,
                            masks_full,
                            reference_bboxes=ref,
                        )

                        frame_data_masks = []
                        frame_data_ids = []
                        frame_data_scores = []
                        frame_valid_bboxes = {}
                        for lid, mask in zip(obj_ids_out, masks_full):
                            lid = id_assoc.local_id_to_int(lid)
                            gid = id_assoc.get_global_id(seg_idx, lid)
                            if gid is None:
                                continue
                            bbox = id_assoc.valid_bbox(mask)
                            if bbox is None:
                                continue
                            frame_data_masks.append(mask)
                            frame_data_ids.append(gid)
                            frame_data_scores.append(local_obj_scores.get(int(lid), np.nan))
                            frame_valid_bboxes[int(gid)] = bbox
                        for gid, bbox in frame_valid_bboxes.items():
                            if bbox is not None:
                                seg_last_bboxes_local[int(gid)] = bbox
                                id_assoc.observe(int(gid), bbox, seg_idx)

                        np.savez_compressed(
                            os.path.join(temp_cache_dir, f"{frame_idx:06d}.npz"),
                            masks=np.array(frame_data_masks) if frame_data_masks else np.array([]),
                            track_ids=np.array(frame_data_ids, dtype=np.int64) if frame_data_ids else np.array([], dtype=np.int64),
                            scores=np.array(frame_data_scores, dtype=np.float32) if frame_data_scores else np.array([]),
                        )
                        count += 1
                        if pbar is not None:
                            pbar.update(1)

                        # Prune SAM2 non-cond outputs older than num_maskmem window
                        # to prevent unbounded CPU RAM growth on long videos.
                        self._sam2_prune_memory(inference_state, frame_idx)

                        # Periodic GPU cache flush — mirrors BL2/3/4 cadence.
                        if count % 20 == 0:
                            self._cleanup_gpu()

                id_assoc.end_segment(seg_idx, seg_last_bboxes_local)
                self._cleanup_gpu()

        finally:
            if pbar is not None:
                pbar.close()
            if _owns_frame_dir:
                import shutil
                shutil.rmtree(frame_dir, ignore_errors=True)

        elapsed = time.time() - t0
        self._log(f"Tracking done: {count} frames in {elapsed:.1f}s ({count/elapsed:.1f} fps)")
        return temp_cache_dir, total_frames, fps, (W, H)
