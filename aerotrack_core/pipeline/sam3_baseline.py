from aerotrack_core.pipeline.ov_tracker_base import BaseOVTracker


class SAM3BaselineTracker(BaseOVTracker):
    """Baseline 1 (Lower Bound): Pure SAM3 for UAV open-vocabulary tracking.

    No external detector is used. SAM3's built-in text-driven auto-detector
    initializes targets on the first frame, and its memory-based propagation
    handles all subsequent frames.

    This serves as the lower-bound baseline to measure how much gain the
    re-localization module (Baselines 2 & 3) brings in UAV scenarios where
    objects undergo frequent occlusion and scale change.

    Typical observed behaviour on VisDrone:
        - SAM3 can initialize 5-20 cars from a text prompt on frame 0.
        - Tracking quality degrades beyond ~100 frames due to no re-detection.
        - GPU memory stabilizes after ~num_maskmem (7) frames are buffered.
    """

    def _load_detector(self):
        # No external detector needed; SAM3's internal text-driven detector
        # handles initialization via the add_prompt("text=...") API.
        pass

    def detect_and_relocate(self, frame, text_prompt):
        # Returning (None, None) tells BaseOVTracker to fall back to SAM3's
        # native text prompt path instead of injecting external bounding boxes.
        return None, None

