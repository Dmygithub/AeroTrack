"""Incremental warm-restart tracker base class (BL7/BL8/BL9 shared architecture).

Warm-restart vs BL2/BL3 hard-restart:
  BL2/BL3: reset_session → re-detect ALL objects → re-inject ALL → new IDs each segment.
  BL7/BL8/BL9: reset_session → re-inject EXISTING tracks (from saved masks, IDs unchanged)
           → detect-only NEW objects → inject only new ones with new IDs.

Result: track IDs are globally continuous across the entire video, and each
re-detection step only costs computation proportional to *new* objects, not all
objects. Memory is bounded to one segment's worth of SAM3 state (same as BL2/BL3).

Subclasses only need to implement ``detect_on_keyframe`` and ``_load_detector``.
"""

import os
import time
import tempfile
import glob as _glob

import cv2
import torch
import numpy as np
from abc import abstractmethod

from aerotrack_core.pipeline.ov_tracker_base import BaseOVTracker


class IncrementalOVTracker(BaseOVTracker):
    """Base class for warm-restart incremental trackers (BL7/BL8/BL9).

    Core invariant per segment:
      1. reset_session()                            (clears SAM3 state, memory-bounded)
      2. Re-inject all active tracks using stored mask bboxes (IDs unchanged)
      3. detect_on_keyframe() → candidate bboxes
      4. Match candidates against active track bboxes via IoU
      5. Inject only unmatched candidates as *new* tracks
      6. Batch-propagate the full segment
      7. Update active_tracks with the last frame's masks/bboxes
    """

    relocation_interval: int  # must be set by subclass

    IOU_MATCH_THRESH = 0.3

    # ------------------------------------------------------------------
    # Abstract detector interface
    # ------------------------------------------------------------------

    @abstractmethod
    def detect_on_keyframe(self, frame, frame_idx, text_prompt, inference_state):
        """Run the open-vocabulary detector on a single key frame.

        Args:
            frame: BGR numpy image (H, W, 3).
            frame_idx: The integer index of this key frame in the video.
            text_prompt: Open-vocabulary category text.
            inference_state: Current SAM3 inference state.

        Returns:
            bboxes: list of [x1, y1, x2, y2] pixel coordinates.
            confidences: list of float scores.
        """
        ...

    # ------------------------------------------------------------------
    # Bbox / IoU helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_to_bbox(mask):
        """[x1, y1, x2, y2] from a 2-D boolean mask, or None if empty."""
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    @staticmethod
    def _iou_bbox(a, b):
        """IoU between two [x1, y1, x2, y2] boxes."""
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _match_new_to_existing(self, new_bboxes, existing_bboxes):
        """Greedy IoU matching: return list of new_bboxes with no match above threshold."""
        thresh = getattr(self, "match_iou_thr", getattr(self, "IOU_MATCH_THRESH", 0.3))
        unmatched = []
        for nb in new_bboxes:
            matched = any(
                self._iou_bbox(nb, eb) >= thresh
                for eb in existing_bboxes
                if eb is not None
            )
            if not matched:
                unmatched.append(nb)
        return unmatched

    # ------------------------------------------------------------------
    # Track management helpers
    # ------------------------------------------------------------------

    def _read_frame(self, video_path, frames_list, frame_idx):
        """Read a single frame as BGR numpy array."""
        if frames_list is not None:
            return cv2.imread(frames_list[frame_idx])
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()
        cap.release()
        return frame




    def _inject_with_box(self, session_id, frame_idx, obj_id, bbox, W, H):
        """Inject one object bbox into SAM3 via add_tracker_new_points."""
        x1, y1, x2, y2 = bbox
        x1 = max(0.0, float(x1))
        y1 = max(0.0, float(y1))
        x2 = min(float(x2), float(W))
        y2 = min(float(y2), float(H))

        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            return

        box_norm = torch.tensor(
            [x1 / W, y1 / H, x2 / W, y2 / H], dtype=torch.float32
        )
        inference_state = self.video_predictor._ALL_INFERENCE_STATES[session_id]["state"]
        model = getattr(self.video_predictor, "model", None)
        if model is None:
            return

        if "cached_frame_outputs" not in inference_state:
            inference_state["cached_frame_outputs"] = {}
        if frame_idx not in inference_state["cached_frame_outputs"]:
            inference_state["cached_frame_outputs"][frame_idx] = {}

        try:
            model.add_tracker_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box_norm,
                rel_coordinates=True,
            )
        except Exception as e:
            # Skip failed injection on very high-resolution frames.
            _log = getattr(self, "_log", print)
            _log(f"[WARN] add_tracker_new_points failed for obj {obj_id} "
                 f"({W}x{H}): {e}")
            return

        rank0_meta = inference_state.get("tracker_metadata", {}).get("rank0_metadata")
        if rank0_meta is not None:
            init_ka = getattr(model, "init_trk_keep_alive", 30)
            rank0_meta["obj_first_frame_idx"][obj_id] = frame_idx
            rank0_meta["trk_keep_alive"][obj_id] = init_ka


    # ------------------------------------------------------------------
    # Main tracking loop
    # ------------------------------------------------------------------

    def run_tracking(self, video_path, text_prompt, output_dir=None, prompt_frame_idx=0):
        """Warm-restart incremental tracking loop.

        Returns (temp_cache_dir, total_frames, fps, (W, H)).
        """
        self._text_prompt = text_prompt
        self._cleanup_gpu()

        # --- resolve video source ---
        if os.path.isdir(video_path):
            frames_list = []
            for ext in ("jpg", "jpeg", "png"):
                frames_list.extend(_glob.glob(os.path.join(video_path, f"*.{ext}")))
            frames_list = sorted(set(frames_list))
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

        H_vid, W_vid = first_frame_im.shape[:2]

        reloc_interval = getattr(self, "relocation_interval", 0)
        if reloc_interval > 0:
            key_frames = list(range(prompt_frame_idx, total_frames, reloc_interval))
        else:
            key_frames = [prompt_frame_idx]

        temp_cache_dir = tempfile.mkdtemp(prefix="tracker_cache_")
        t0 = time.time()
        count = 0

        # --- progress bar ---
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

        def _log(msg):
            if not self.verbose:
                return
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg, flush=True)

        def _save_outputs(frame_idx, outputs):
            masks = outputs.get("out_binary_masks")
            track_ids = outputs.get("out_obj_ids")
            if masks is not None:
                masks = masks.cpu().numpy().astype(bool) if hasattr(masks, "cpu") else masks.astype(bool)
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy() if hasattr(track_ids, "cpu") else track_ids

            # Map SAM3 local obj_ids → global track IDs for color stability.
            if track_ids is not None and len(track_ids) > 0:
                global_ids = np.array(
                    [obj_id_to_track.get(int(oid), int(oid)) for oid in track_ids],
                    dtype=track_ids.dtype,
                )
            else:
                global_ids = track_ids if track_ids is not None else np.array([])

            np.savez_compressed(
                os.path.join(temp_cache_dir, f"{frame_idx:06d}.npz"),
                masks=masks if masks is not None else np.array([]),
                track_ids=global_ids,
            )
            return masks, track_ids  # return original SAM3 obj_ids for internal use

        # -------------------------------------------------------------------
        # Track state across segments:
        #   active_tracks[global_track_id] = {
        #       'bbox': [x1,y1,x2,y2],   # last-known bbox (for IoU matching)
        #       'mask': np.ndarray H×W,  # last-known binary mask (for re-injection)
        #   }
        # obj_id_to_track[obj_id_in_session] = global_track_id
        # -------------------------------------------------------------------
        active_tracks = {}          # global_track_id → {bbox, mask}
        obj_id_to_track = {}        # SAM3 obj_id (in current session) → global_track_id
        next_track_id = 0

        # --- start the SAM3 session ---
        _log(f"Starting SAM3 session for: {video_path}")
        result = self.video_predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = result["session_id"]

        try:
            for seg_idx, key_frame in enumerate(key_frames):
                seg_end = key_frames[seg_idx + 1] if seg_idx + 1 < len(key_frames) else total_frames
                seg_len = seg_end - key_frame

                key_img = self._read_frame(video_path, frames_list, key_frame)

                # ----------------------------------------------------------
                # Step A: Reset session (if not first segment)
                # ----------------------------------------------------------
                if seg_idx > 0:
                    try:
                        _state_entry = self.video_predictor._ALL_INFERENCE_STATES.get(session_id, {})
                        _istate = _state_entry.get("state", {})
                        _fc = _istate.get("feature_cache")
                        if _fc:
                            _fc.clear()
                        for _ts in _istate.get("tracker_inference_states", []):
                            _od = _ts.get("output_dict", {})
                            _od.get("non_cond_frame_outputs", {}).clear()
                            for _pobj in _ts.get("output_dict_per_obj", {}).values():
                                _pobj.get("non_cond_frame_outputs", {}).clear()
                    except Exception:
                        pass
                    self.video_predictor.handle_request(
                        request=dict(type="reset_session", session_id=session_id)
                    )
                    self._cleanup_gpu()

                # Rebuild obj_id_to_track for this segment.
                obj_id_to_track = {}

                # ----------------------------------------------------------
                # Step B: Detect new objects on this key frame
                # (Detect FIRST to avoid temporary text-detections resetting 
                # previously injected existing tracks)
                # ----------------------------------------------------------
                _log(f"\n[DEBUG] === Key Frame {key_frame} ===")
                inference_state = self.video_predictor._ALL_INFERENCE_STATES[session_id]["state"]
                det_bboxes, det_confs = self.detect_on_keyframe(key_img, key_frame, text_prompt, inference_state)
                _log(f"[DEBUG] detect_on_keyframe returned {len(det_bboxes)} bboxes.")

                # Step C: Match detections to active tracks before injecting new ones.
                _min_conf = getattr(self, 'inject_min_conf', 0.0)
                if _min_conf > 0.0 and det_confs:
                    _before = len(det_bboxes)
                    det_bboxes = [b for b, c in zip(det_bboxes, det_confs) if c >= _min_conf]
                    det_confs  = [c for c in det_confs if c >= _min_conf]
                    _dropped = _before - len(det_bboxes)
                    if _dropped > 0:
                        _log(f'[FILTER] dropped {_dropped}/{_before} detections (conf < {_min_conf})')

                thresh = getattr(self, "match_iou_thr", getattr(self, "IOU_MATCH_THRESH", 0.3))
                new_bboxes = []
                
                matched_tids = set()
                
                for nb in det_bboxes:
                    best_iou = 0
                    best_tid = None
                    for tid, tstate in active_tracks.items():
                        if tid in matched_tids:
                            continue
                        eb = tstate.get("bbox")
                        if eb is not None:
                            iou = self._iou_bbox(nb, eb)
                            if iou > best_iou:
                                best_iou = iou
                                best_tid = tid

                    if best_iou >= thresh:
                        _log(f"[DEBUG] ✨ Track {best_tid} matched to G-DINO detection (IoU: {best_iou:.2f}), replacing old bbox.")
                        active_tracks[best_tid]["bbox"] = nb
                        active_tracks[best_tid]["missing_segments"] = 0
                        matched_tids.add(best_tid)
                    else:
                        new_bboxes.append(nb)

                _log(f"[DEBUG] After IoU match: existing={len(active_tracks)}, matched={len(matched_tids)}, remaining_new={len(new_bboxes)}")

                # Limit new injections to respect max_objects cap.
                max_obj = getattr(self, "max_objects", None)
                if max_obj and max_obj > 0:
                    remaining_slots = max_obj - len(active_tracks)
                    if len(new_bboxes) > remaining_slots:
                        _log(f"[DEBUG] Cap limit hit! Active={len(active_tracks)}, remaining slots={remaining_slots}. Cropping {len(new_bboxes)} to {max(0, remaining_slots)}.")
                    new_bboxes = new_bboxes[:max(0, remaining_slots)]

                # ----------------------------------------------------------
                # Step D: Re-inject ALL active tracks using stored (and updated) bboxes
                #         (keeps their global track IDs alive across the reset)
                # ----------------------------------------------------------
                reinject_next_obj_id = 0
                for tid, tstate in list(active_tracks.items()):
                    bbox = tstate.get("bbox")
                    if bbox is None:
                        continue
                    obj_id = reinject_next_obj_id
                    reinject_next_obj_id += 1
                    try:
                        self._inject_with_box(session_id, key_frame, obj_id, bbox, W_vid, H_vid)
                        obj_id_to_track[obj_id] = tid
                    except Exception as e:
                        _log(f"[WARN] Re-injection of track {tid} failed: {e}")

                # ----------------------------------------------------------
                # Step E: Inject new objects
                # ----------------------------------------------------------
                for bbox in new_bboxes:
                    obj_id = reinject_next_obj_id
                    reinject_next_obj_id += 1
                    try:
                        self._inject_with_box(session_id, key_frame, obj_id, bbox, W_vid, H_vid)
                        active_tracks[next_track_id] = {"bbox": bbox, "mask": None, "missing_segments": 0}
                        obj_id_to_track[obj_id] = next_track_id
                        next_track_id += 1
                    except Exception as e:
                        _log(f"[WARN] Injection of new track failed: {e}")

                if seg_idx == 0:
                    _log(f"[Key {key_frame}] Initial: {len(active_tracks)} tracks injected")
                else:
                    _log(
                        f"[Key {key_frame}] Warm-restart: {len(obj_id_to_track) - len(new_bboxes)} re-injected, "
                        f"{len(new_bboxes)} new, total active={len(active_tracks)}"
                    )

                # ----------------------------------------------------------
                # Step E2: Set VG text prompt for detection-to-track matching
                # ----------------------------------------------------------
                if obj_id_to_track:
                    inference_state = self.video_predictor._ALL_INFERENCE_STATES[session_id]["state"]
                    inference_state["text_prompt"] = text_prompt
                    if hasattr(inference_state.get("input_batch"), "find_text_batch"):
                        inference_state["input_batch"].find_text_batch[0] = text_prompt
                    # Force propagation_full on the next propagate_in_video call
                    # so that all newly injected objects get proper initial masks.
                    if "action_history" in inference_state:
                        inference_state["action_history"].clear()

                if not obj_id_to_track:
                    _log(f"[Key {key_frame}] No active tracks, skipping segment")
                    if pbar is not None:
                        pbar.update(seg_len)
                    count += seg_len
                    continue

                # ----------------------------------------------------------
                # Step F: Batch-propagate the full segment
                # ----------------------------------------------------------
                _log(f"[BL7/8] Propagating frames {key_frame} → {seg_end - 1}")
                last_masks = None
                last_obj_ids = None

                inference_state = self.video_predictor._ALL_INFERENCE_STATES[session_id]["state"]
                with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                    for sr in self.video_predictor.handle_stream_request(
                        request=dict(
                            type="propagate_in_video",
                            session_id=session_id,
                            propagation_direction="forward",
                            start_frame_index=key_frame,
                            max_frame_num_to_track=seg_len,
                        )
                    ):
                        fidx = sr["frame_index"]
                        last_masks, last_obj_ids = _save_outputs(fidx, sr["outputs"])
                        count += 1
                        if pbar is not None:
                            pbar.update(1)

                        # Prune stale tracker memory to cap CPU RAM growth
                        self._prune_stale_tracker_memory(inference_state, fidx)

                        # Periodic GPU cache flush
                        if count % 20 == 0:
                            self._cleanup_gpu()

                # ----------------------------------------------------------
                # Step G: Update active_tracks with last frame's masks/bboxes
                # ----------------------------------------------------------
                for tstate in active_tracks.values():
                    tstate["missing_segments"] = tstate.get("missing_segments", 0) + 1
                    tstate["mask"] = None

                has_valid_last = (
                    last_masks is not None and
                    last_obj_ids is not None and
                    getattr(last_masks, "size", 1) > 0 and
                    len(last_obj_ids) > 0
                )
                if has_valid_last:
                    for i, sam_obj_id in enumerate(last_obj_ids):
                        sam_obj_id = int(sam_obj_id)
                        tid = obj_id_to_track.get(sam_obj_id)
                        if tid is None or tid not in active_tracks:
                            continue
                        if i >= len(last_masks):
                            continue
                        mask_2d = last_masks[i].squeeze()
                        bbox = self._mask_to_bbox(mask_2d)
                        if bbox is not None:
                            active_tracks[tid]["bbox"] = bbox
                            active_tracks[tid]["mask"] = mask_2d
                            active_tracks[tid]["missing_segments"] = 0

                max_missing = getattr(self, "max_missing_segments", 1)
                gone = [tid for tid, s in active_tracks.items() if s.get("missing_segments", 0) > max_missing]
                for tid in gone:
                    del active_tracks[tid]
                    _log(f"[DEBUG] ❌ Track {tid} removed after exceeding cooldown limit.")

                self._cleanup_gpu()

            self._cleanup_gpu()
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

        _log(f"Tracking completed in {total_time:.1f}s ({count} frames).")
        _log(f"Performance Stats | Avg FPS: {avg_fps:.2f} | Peak GPU Mem: {peak_mem_mb:.2f} MB")
        return temp_cache_dir, total_frames, fps, (W_vid, H_vid)
