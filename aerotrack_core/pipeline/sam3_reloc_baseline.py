from aerotrack_core.pipeline.ov_tracker_base import BaseOVTracker


class SAM3RelocTracker(BaseOVTracker):
    """Baseline 2 scaffold: SAM3 with periodic hard-interval re-localization.

    Implements the two-stage framework:
        - Global Static Localization: SAM3's own text-driven detector fires on
          every key frame (spaced `relocation_interval` frames apart).
        - Dynamic Mask Propagation: SAM3 propagates masks within each segment.

    This is an intermediate baseline that uses only SAM3 (no external detector)
    but gains the ability to discover new instances at each key frame by
    restarting the session with a fresh text prompt. It serves as a stepping
    stone before plugging in YOLO-World / GroundingDINO as the localization
    module in Baselines 3 / 2.

    Compared with Baseline 1 (pure SAM3, single pass):
        + New instances entering after the first frame can be discovered
        + Tracks that drifted badly are reset at each key frame
        + Lifecycle ID association can reconnect objects across segment resets
        - Slightly higher latency due to periodic session restarts
        - Without lifecycle ID association, hard resets would re-assign IDs

    Args:
        sam3_checkpoint: Path to SAM3 model weights.
        relocation_interval: Number of frames between consecutive key frames.
            Smaller values mean more frequent re-detection and higher compute cost.
        max_objects: Cap on simultaneously tracked instances (passed to SAM3).
        new_det_thresh: Detection confidence threshold for new track creation.
        max_trk_keep_alive: Frames a lost track is kept alive before removal.
    """

    def __init__(
        self,
        sam3_checkpoint,
        relocation_interval=30,
        max_objects=20,
        new_det_thresh=0.85,
        max_trk_keep_alive=8,
        match_iou_thr=0.3,
        max_missing_segments=1,
        max_mask_area_ratio=0.5,
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
        self.relocation_interval = relocation_interval

    def _load_detector(self):
        # SAM3's internal text-driven detector is used for localization.
        # No external detector is loaded at this stage.
        pass

    def detect_and_relocate(self, frame, text_prompt):
        # Returning (None, None) signals BaseOVTracker to use SAM3's native
        # text-prompt path, triggering its internal auto-detector on the key frame.
        return None, None
