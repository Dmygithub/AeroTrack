import os
import gc
import cv2
import time
import uuid
import torch
import numpy as np
from abc import ABC, abstractmethod

# Suppress warnings and default logs for cleaner output
os.environ["LOG_LEVEL"] = "ERROR"
import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

from aerotrack_core.models.sam3.model.sam3_video_predictor import Sam3VideoPredictor
from aerotrack_core.pipeline.id_association import build_video_id_association

class BaseOVTracker(ABC):
    """
    Base Open-Vocabulary Tracker for UAV-OVTrack.
    Defines the dual-state pipeline: Static Re-localization + Dynamic Mask Propagation.
    """
    def __init__(
        self,
        sam3_checkpoint,
        max_objects=20,
        new_det_thresh=0.85,
        max_trk_keep_alive=8,
        match_iou_thr=0.3,
        max_missing_segments=1,
        max_mask_area_ratio=0.5,
        use_track_lifecycle=False,
        lost_track_ttl_segments=2,
        lifecycle_match_score_thr=0.45,
        lifecycle_center_gate=2.5,
        lifecycle_area_ratio_min=0.35,
        lifecycle_area_ratio_max=2.8,
        verbose=False,
        show_progress=True,
    ):
        self.sam3_checkpoint = sam3_checkpoint
        self.max_objects = max_objects
        self.new_det_thresh = new_det_thresh
        self.max_trk_keep_alive = max_trk_keep_alive
        self.match_iou_thr = match_iou_thr
        self.max_missing_segments = max_missing_segments
        self.max_mask_area_ratio = max_mask_area_ratio
        self.use_track_lifecycle = use_track_lifecycle
        self.lost_track_ttl_segments = lost_track_ttl_segments
        self.lifecycle_match_score_thr = lifecycle_match_score_thr
        self.lifecycle_center_gate = lifecycle_center_gate
        self.lifecycle_area_ratio_min = lifecycle_area_ratio_min
        self.lifecycle_area_ratio_max = lifecycle_area_ratio_max
        self.video_predictor = None
        self.verbose = verbose
        self.show_progress = show_progress

        # Prefer a single, top-level progress bar. Internal SAM3 progress bars
        # can be re-enabled by setting SAM3_SHOW_PROGRESS=1.
        if not self.show_progress:
            os.environ["SAM3_SHOW_PROGRESS"] = "0"
        elif "SAM3_SHOW_PROGRESS" not in os.environ:
            os.environ["SAM3_SHOW_PROGRESS"] = "0"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Optimize PyTorch runtime for Ampere+ GPUs
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    def _cleanup_gpu(self):
        gc.collect()
        if torch.cuda.is_available() and not os.environ.get("AEROTRACK_MEMORY_RESERVE"):
            torch.cuda.empty_cache()

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _set_local_obj_score(self, obj_id, score):
        """Record detector confidence for a SAM local object id."""
        if score is None:
            return
        try:
            score = float(score)
        except (TypeError, ValueError):
            return
        if not np.isfinite(score):
            return
        if not hasattr(self, "_local_obj_scores"):
            self._local_obj_scores = {}
        self._local_obj_scores[int(obj_id)] = float(np.clip(score, 0.0, 1.0))

    def load_models(self):
        """Loads SAM3 and any required detectors"""
        self._cleanup_gpu()
        self._log(f"Loading SAM3 Predictor from {self.sam3_checkpoint}...")
        self.video_predictor = Sam3VideoPredictor(
            checkpoint_path=self.sam3_checkpoint
        )
        m = self.video_predictor.model if hasattr(self.video_predictor, "model") else None
        if m is not None:
            if hasattr(m, "max_num_objects"):
                m.max_num_objects = self.max_objects
            if hasattr(m, "new_det_thresh"):
                m.new_det_thresh = self.new_det_thresh
            if hasattr(m, "max_trk_keep_alive"):
                m.max_trk_keep_alive = self.max_trk_keep_alive
        
        self._load_detector()
        self._log("All models loaded successfully.")

    @abstractmethod
    def _load_detector(self):
        """Implement to load YOLO-World or GroundingDINO."""
        pass

    @abstractmethod
    def detect_and_relocate(self, frame, text_prompt):
        """
        Runs the open-vocabulary detector on the frame.
        Returns:
            bboxes: List of [x1, y1, x2, y2]
            confidences: List of floats
        """
        pass

    def _read_frame(self, video_path, frames_list, frame_idx):
        """Read a single frame from video or image folder."""
        if frames_list is not None:
            return cv2.imread(frames_list[frame_idx])
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, img = cap.read()
        cap.release()
        return img

    def _add_boxes_as_prompt(self, session_id, frame_idx, img, bboxes, obj_ids=None, scores=None):
        """Convert xyxy pixel boxes to prompts and inject into SAM3 session.

        Args:
            session_id: Active SAM3 session ID.
            frame_idx: Frame index for the prompt.
            img: BGR image array used to get H, W for normalization.
            bboxes: List of [x1, y1, x2, y2] pixel-coordinate boxes.
            obj_ids: Optional list of obj_ids aligned with bboxes for refinement.
                If provided, SAM3 will refine existing objects using these ids.
            scores: Optional detector confidences aligned with bboxes.
        """
        H, W = img.shape[:2]

        # Use SAM3's official visual box-prompt path for all box injections.
        # add_prompt(text="visual", boxes=...) is the only supported multi-box init
        # interface. It calls reset_state internally which clears tracker state, but
        # this is acceptable for the first call on a new session. For incremental
        # re-detection during propagation, callers should use add_tracker_new_points
        # directly (see YOLOWorldBaselineTracker._apply_yolo_results).
        if obj_ids is not None:
            raise ValueError("obj_ids refinement requires add_tracker_new_points; "
                             "call _apply_yolo_results instead of _add_boxes_as_prompt.")
        boxes_xywh_norm = [
            [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]
            for x1, y1, x2, y2 in bboxes
        ]
        self.video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                text="visual",
                bounding_boxes=boxes_xywh_norm,
                bounding_box_labels=[1] * len(boxes_xywh_norm),
            )
        )

    def _prune_stale_tracker_memory(self, inference_state, current_frame_idx):
        """Delete stale tracker memories older than the memory window.

        SAM3 tracker states keep both conditioning-frame outputs and
        non-conditioning-frame outputs. In reset-based baselines this is
        naturally bounded by segment resets, but in incremental single-session
        tracking the structures can grow without bound unless they are trimmed.

        This helper does two things:

        1. demote overly old conditioning frames so only the most recent
           `max_cond_frames_in_attn` cond frames remain;
        2. trim non-conditioning outputs older than the `num_maskmem` window.

        Args:
            inference_state: SAM3 session inference state.
            current_frame_idx: Current propagation frame index.
        """
        tracker_states = inference_state.get("tracker_inference_states", [])
        tracker_model = getattr(getattr(self.video_predictor, "model", None), "tracker", None)
        window = getattr(tracker_model, "num_maskmem", 7) if tracker_model is not None else 7
        cutoff = current_frame_idx - window
        for state in tracker_states:
            output_dict = state.get("output_dict", {})
            non_cond = output_dict.get("non_cond_frame_outputs")
            cond = output_dict.get("cond_frame_outputs")

            # Incremental single-session runs can keep adding detector
            # conditioning frames forever. Demote old ones so only the most
            # recent cond frames remain in attention.
            if tracker_model is not None and cond:
                max_cond = getattr(tracker_model, "max_cond_frames_in_attn", -1)
                if max_cond is not None and max_cond != -1:
                    max_cond = max(2, int(max_cond))
                    cond_frames = sorted(cond.keys())
                    frames_to_demote = cond_frames[:-max_cond]
                    for frame_idx in frames_to_demote:
                        for obj_id in list(state.get("obj_ids", [])):
                            try:
                                tracker_model.clear_all_points_in_frame(
                                    state,
                                    frame_idx,
                                    int(obj_id),
                                    need_output=False,
                                )
                            except Exception:
                                # Best-effort trimming only; if a frame is
                                # already partially cleaned, keep going.
                                continue

            if cutoff <= 0:
                continue

            # Preferred path: tracker states store outputs under output_dict.
            if non_cond is not None:
                stale = [k for k in list(non_cond.keys()) if k < cutoff]
                for frame_idx in stale:
                    non_cond.pop(frame_idx, None)
                    consolidated = state.get("consolidated_frame_inds", {})
                    if "non_cond_frame_outputs" in consolidated:
                        consolidated["non_cond_frame_outputs"].discard(frame_idx)
                    frames_already_tracked = state.get("frames_already_tracked", {})
                    frames_already_tracked.pop(frame_idx, None)
                    for obj_output_dict in state.get("output_dict_per_obj", {}).values():
                        obj_output_dict.get("non_cond_frame_outputs", {}).pop(frame_idx, None)
                continue

            # Backward-compatible fallback for older state layouts.
            non_cond_legacy = state.get("non_cond_frame_outputs", {})
            stale = [k for k in list(non_cond_legacy.keys()) if k < cutoff]
            for frame_idx in stale:
                del non_cond_legacy[frame_idx]

        # Prune image-encoder feature cache: dominant memory growth source
        # in single-session runs (~12 MB/frame; 275 frames = ~3.3 GB).
        if cutoff > 0:
            feature_cache = inference_state.get("feature_cache", {})
            for k in [k for k in list(feature_cache.keys()) if isinstance(k, int) and k < cutoff]:
                del feature_cache[k]

    def run_tracking(self, video_path, text_prompt, output_dir=None, prompt_frame_idx=0):
        """Main pipeline: returns (temp_cache_dir, total_frames, fps, (W, H))."""
        import glob as _glob
        import tempfile

        self._cleanup_gpu()

        # --- resolve video source ---
        if os.path.isdir(video_path):
            frames_list = []
            for ext in ("jpg", "jpeg", "png"):
                frames_list.extend(_glob.glob(os.path.join(video_path, f"*.{ext}")))
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
            first_frame_im = cv2.imread(frames_list[0])
            total_frames = len(frames_list)
            fps = 30.0
        else:
            frames_list = None
            cap = cv2.VideoCapture(video_path)
            ret, first_frame_im = cap.read()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            if not ret or first_frame_im is None:
                raise RuntimeError(f"Failed to read the first video frame: {video_path}")

        reloc_interval = getattr(self, "relocation_interval", 0)
        H, W = first_frame_im.shape[:2]
        # Segment boundary indices: [prompt_frame, prompt+interval, prompt+2*interval, ...]
        if reloc_interval > 0:
            key_frames = list(range(prompt_frame_idx, total_frames, reloc_interval))
        else:
            key_frames = [prompt_frame_idx]  # single pass for Baseline 1

        temp_cache_dir = tempfile.mkdtemp(prefix="tracker_cache_")
        t0 = time.time()
        count = 0
        id_assoc = build_video_id_association(self, frame_area=H * W)

        # Create the SAM3 session once and reuse across all segments.
        # `reset_state` clears tracking state but retains the loaded image tensors,
        # avoiding the expensive per-segment frame-loading (~28s/segment for BL2).
        progress_total = max(0, total_frames - prompt_frame_idx)
        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    total=progress_total,
                    desc="Tracking",
                    unit="frame",
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:
                pbar = None

        def _log_progress(msg):
            if not self.verbose:
                return
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg, flush=True)

        _log_progress(f"Initializing SAM3 state for video: {video_path}")
        result = self.video_predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = result["session_id"]
        inference_state = self.video_predictor._ALL_INFERENCE_STATES[session_id]["state"]

        try:
            for seg_idx, key_frame in enumerate(key_frames):
                seg_end = key_frames[seg_idx + 1] if seg_idx + 1 < len(key_frames) else total_frames

                # --- 1. Global Static Localization on key frame ---
                # Reset tracking state only (image tensors are retained in the session).
                if seg_idx > 0:
                    self.video_predictor.handle_request(
                        request=dict(type="reset_session", session_id=session_id)
                    )
                    if hasattr(self, "_next_tracker_obj_id"):
                        self._next_tracker_obj_id = 0
                self._local_obj_scores = {}

                key_img = self._read_frame(video_path, frames_list, key_frame)
                bboxes, det_confs = self.detect_and_relocate(key_img, text_prompt)

                if bboxes is None:
                    # BL1/BL2 path: native SAM3 text prompt
                    _log_progress(f"Detecting '{text_prompt}' on initial frame {key_frame}...")
                    self.video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=key_frame,
                            text=text_prompt,
                        )
                    )
                elif len(bboxes) > 0:
                    _log_progress(f"[Key frame {key_frame}] Injecting {len(bboxes)} boxes from detector")
                    self._add_boxes_as_prompt(
                        session_id, key_frame, key_img, bboxes, scores=det_confs
                    )
                else:
                    _log_progress(f"[Key frame {key_frame}] Detector returned no boxes; skipping segment")
                    id_assoc.end_segment(seg_idx, {})
                    if pbar is not None:
                        pbar.update(max(0, seg_end - key_frame))
                    continue

                # --- 2. Dynamic Mask Propagation for this segment ---
                # Collect final valid boxes for lifecycle-aware ID association.
                seg_last_bboxes_local = {}
                _log_progress(f"Propagating frames {key_frame} → {seg_end - 1}...")
                seg_frames_to_track = seg_end - key_frame

                with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                    for stream_response in self.video_predictor.handle_stream_request(
                        request=dict(
                            type="propagate_in_video",
                            session_id=session_id,
                            propagation_direction="forward",
                            start_frame_index=key_frame,
                            max_frame_num_to_track=seg_frames_to_track,
                        )
                    ):
                        frame_idx = stream_response["frame_index"]
                        out = stream_response["outputs"]

                        masks = out.get("out_binary_masks")
                        track_ids = out.get("out_obj_ids")

                        if masks is not None:
                            masks = (
                                masks.cpu().numpy().astype(bool)
                                if hasattr(masks, "cpu")
                                else masks.astype(bool)
                            )
                        if track_ids is not None:
                            track_ids = (
                                track_ids.cpu().numpy()
                                if hasattr(track_ids, "cpu")
                                else track_ids
                            )

                        # Remap local obj_ids to video-level IDs.
                        frame_scores = None
                        frame_valid_bboxes = {}
                        if track_ids is not None and len(track_ids) > 0 and masks is not None:
                            local_track_ids = np.array(track_ids, copy=True)
                            track_ids = id_assoc.assign_frame(
                                seg_idx,
                                track_ids,
                                masks,
                                reference_bboxes=id_assoc.previous_bboxes(),
                            )
                            local_scores = getattr(self, "_local_obj_scores", {})
                            if local_scores:
                                frame_scores = np.array(
                                    [
                                        local_scores.get(int(lid), np.nan)
                                        for lid in local_track_ids
                                    ],
                                    dtype=np.float32,
                                )

                            # Filter out full-frame masks (SAM3 propagation degradation).
                            keep = []
                            for idx, (gid, mask) in enumerate(zip(track_ids, masks)):
                                if int(gid) < 0:
                                    continue
                                m = mask[0] if mask.ndim == 3 else mask
                                bbox = id_assoc.valid_bbox(m)
                                if bbox is not None:
                                    keep.append(idx)
                                    frame_valid_bboxes[int(gid)] = bbox
                            if len(keep) == 0:
                                masks = None
                                track_ids = np.array([], dtype=np.int64)
                                frame_scores = None
                            elif len(keep) < len(track_ids):
                                keep_arr = np.array(keep, dtype=np.intp)
                                masks = masks[keep_arr]
                                track_ids = track_ids[keep_arr]
                                if frame_scores is not None:
                                    frame_scores = frame_scores[keep_arr]
                        for gid, bbox in frame_valid_bboxes.items():
                            if bbox is not None:
                                seg_last_bboxes_local[int(gid)] = bbox
                                id_assoc.observe(int(gid), bbox, seg_idx)

                        score_payload = {}
                        if frame_scores is not None:
                            score_payload["scores"] = frame_scores
                        np.savez_compressed(
                            os.path.join(temp_cache_dir, f"{frame_idx:06d}.npz"),
                            masks=masks if masks is not None else np.array([]),
                            track_ids=track_ids if track_ids is not None else np.array([], dtype=np.int64),
                            **score_payload,
                        )
                        count += 1
                        if pbar is not None:
                            pbar.update(1)

                        # Prune stale tracker memory to cap CPU RAM growth
                        self._prune_stale_tracker_memory(inference_state, frame_idx)

                        # Periodic GPU cache flush
                        if count % 20 == 0:
                            self._cleanup_gpu()

                        # --- Extension point B: Adaptive Confidence-Driven re-localization ---
                        # Baseline 1 & 2 (interval-only) do NOT set adaptive_trigger.
                        if hasattr(self, "adaptive_trigger"):
                            if self.adaptive_trigger.evaluate(out):
                                _log_progress(
                                    f"[Adaptive Wake-Up] Semantic drift at frame {frame_idx}"
                                )
                                wake_img = self._read_frame(video_path, frames_list, frame_idx)
                                wake_bboxes, wake_confs = self.detect_and_relocate(wake_img, text_prompt)
                                if wake_bboxes is not None and len(wake_bboxes) > 0:
                                    self._add_boxes_as_prompt(
                                        session_id, frame_idx, wake_img, wake_bboxes,
                                        scores=wake_confs,
                                    )
                                self.adaptive_trigger.reset()

                self._cleanup_gpu()
                id_assoc.end_segment(seg_idx, seg_last_bboxes_local)

        finally:
            try:
                self.video_predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass
            if pbar is not None:
                pbar.close()

        total_time = time.time() - t0
        avg_fps = count / total_time if total_time > 0 else 0
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        
        _log_progress(f"Tracking completed in {total_time:.1f}s ({count} frames).")
        _log_progress(f"Performance Stats | Avg FPS: {avg_fps:.2f} | Peak GPU Mem: {peak_mem_mb:.2f} MB")
        return temp_cache_dir, total_frames, fps, (first_frame_im.shape[1], first_frame_im.shape[0])
