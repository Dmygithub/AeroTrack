import numpy as np

class AdaptiveConfidenceTrigger:
    """
    CVPR Core Contribution: Adaptive Confidence-Driven Mechanism.
    Analyzes the 'state' of the SAM3 tracker (e.g. area variation, temporal mask IoU)
    to dynamically determine when the tracker has lost the target and needs to wake up
    the open-vocabulary detector.
    
    Implements a 'Decay-Wakeup-Reset' state machine.
    """
    def __init__(self, iou_threshold=0.4, area_change_threshold=2.0, cooldown_frames=10):
        self.iou_threshold = iou_threshold
        self.area_change_threshold = area_change_threshold
        self.cooldown_frames = cooldown_frames
        
        self.reset()

    def reset(self):
        """Resets the state machine after a successful re-localization."""
        self.prev_masks = None
        self.prev_areas = None
        self.frames_since_last_reloc = 0
        self.drift_score = 0.0

    def evaluate(self, sam3_output: dict) -> bool:
        """
        Evaluates the current frame's output to determine if a re-localization is needed.
        sam3_output: The 'outputs' dictionary from SAM3 stream_response.
            Requires 'out_binary_masks' and 'out_obj_ids'.
        """
        self.frames_since_last_reloc += 1
        
        if self.frames_since_last_reloc < self.cooldown_frames:
            # Enforce cooldown to prevent continuous expensive detection
            self._update_state(sam3_output)
            return False
            
        masks = sam3_output.get("out_binary_masks")
        if masks is None:
            return False
            
        # Convert to CPU bool numpy array
        masks = masks.cpu().numpy().astype(bool) if hasattr(masks, 'cpu') else masks.astype(bool)
        
        # If tracker lost all objects completely (empty mask)
        current_areas = masks.sum(axis=(1, 2) if masks.ndim == 3 else (0, 1))
        if np.all(current_areas == 0):
            print("[AdaptiveTrigger] Target completely lost (Area 0).")
            return True
            
        need_wakeup = False
        
        if self.prev_masks is not None and self.prev_masks.ndim == masks.ndim:
            # Object count may change between frames (hotstart add/remove).
            # Compare only the overlapping subset; treat any count change as
            # additional drift signal.
            n_cur, n_prev = masks.shape[0], self.prev_masks.shape[0]
            n_common = min(n_cur, n_prev)
            if n_common == 0:
                self._update_state(sam3_output)
                return True
            m_cur = masks[:n_common]
            m_prev = self.prev_masks[:n_common]
            prev_areas_common = self.prev_areas[:n_common]

            # Temporal IoU Analysis
            sum_ax = (1, 2) if m_cur.ndim == 3 else (0, 1)
            intersection = np.logical_and(m_cur, m_prev).sum(axis=sum_ax)
            union = np.logical_or(m_cur, m_prev).sum(axis=sum_ax)
            ious = np.where(union > 0, intersection / union, 0)

            # Area Variation Analysis
            cur_areas_common = current_areas[:n_common]
            area_ratio = np.where(prev_areas_common > 0, cur_areas_common / prev_areas_common, 0)
            
            # Condition 1: Sudden massive area change (explosion or shrinking)
            # Condition 2: Huge spatial jump (low temporal IoU)
            max_drift = 0
            for iou, ratio in zip(np.atleast_1d(ious), np.atleast_1d(area_ratio)):
                if iou < self.iou_threshold:
                    max_drift += 1
                if ratio > self.area_change_threshold or ratio < (1.0 / self.area_change_threshold):
                    max_drift += 1
                    
            if max_drift > 0:
                self.drift_score += max_drift
            else:
                self.drift_score = max(0.0, self.drift_score - 0.5) # Decay
                
            if self.drift_score >= 2.0: # State Machine Trigger
                need_wakeup = True

        self._update_state(sam3_output)
        return need_wakeup

    def _update_state(self, sam3_output):
        masks = sam3_output.get("out_binary_masks")
        if masks is not None:
            self.prev_masks = masks.cpu().numpy().astype(bool) if hasattr(masks, 'cpu') else masks.astype(bool)
            self.prev_areas = self.prev_masks.sum(axis=(1, 2) if self.prev_masks.ndim == 3 else (0, 1))
