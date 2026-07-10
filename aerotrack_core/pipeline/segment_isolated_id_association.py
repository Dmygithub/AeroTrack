"""Segment-isolated global ID assignment for w/o LIA ablation.

Each hard-restart segment assigns fresh video-level ids without reading any
state from previous segments.  Segment-internal local ids still map stably
within the same segment.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from aerotrack_core.pipeline.id_association import _to_int_id, mask_to_bbox
from aerotrack_core.pipeline.track_lifecycle import _float_range, _int_range


class SegmentIsolatedIDAssociation:
    """Map segment-local ids to fresh global ids with no cross-segment linking."""

    def __init__(
        self,
        *,
        max_mask_area_ratio: float = 0.5,
        frame_area: int,
    ):
        self.max_mask_area_ratio = _float_range(
            "max_mask_area_ratio",
            max_mask_area_ratio,
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        self.frame_area = _int_range("frame_area", frame_area, lower=1)
        self.id_remap: Dict[Tuple[int, int], int] = {}
        self.next_global_id = 0

    @classmethod
    def from_tracker(cls, tracker, *, frame_area: int) -> "SegmentIsolatedIDAssociation":
        """Build an ablation association object from tracker runtime settings."""
        return cls(
            max_mask_area_ratio=getattr(tracker, "max_mask_area_ratio", 0.5),
            frame_area=frame_area,
        )

    @property
    def enabled(self) -> bool:
        """Return False so callers can distinguish ablation mode."""
        return False

    def valid_bbox(self, mask) -> Optional[List[int]]:
        """Return bbox only for non-empty, non-degenerated masks."""
        if mask is None:
            return None
        m = mask[0] if getattr(mask, "ndim", 0) == 3 else mask
        mask_area = int(np.asarray(m).sum())
        if mask_area <= 0:
            return None
        if mask_area > self.frame_area * self.max_mask_area_ratio:
            return None
        bbox = mask_to_bbox(m)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        if bbox_area > self.frame_area * self.max_mask_area_ratio:
            return None
        return bbox

    def _used_global_ids(self) -> set:
        """Return all global ids already assigned in this video."""
        return {int(gid) for gid in self.id_remap.values()}

    def _advance_next_global_id_past(self, used_gids: set) -> None:
        """Keep newly allocated ids from colliding with existing ids."""
        if used_gids:
            self.next_global_id = max(
                int(self.next_global_id),
                max(int(gid) for gid in used_gids) + 1,
            )

    def _result_array(
        self,
        local_ids: Iterable[int],
        keys: Iterable[Tuple[int, int]],
        current_valid_lids: set,
        dtype,
    ) -> np.ndarray:
        """Return mapped gids, using -1 for masks invalid in this frame."""
        return np.array(
            [
                self.id_remap.get(key, -1)
                if int(lid) in current_valid_lids
                else -1
                for lid, key in zip(local_ids, keys)
            ],
            dtype=dtype,
        )

    def assign_frame(
        self,
        segment_idx: int,
        local_ids: Iterable[int],
        masks_this_frame: Iterable[object],
        *,
        reference_bboxes: Optional[Dict[int, object]] = None,
    ) -> np.ndarray:
        """Assign fresh global ids for unseen segment-local ids.

        Args:
            segment_idx: Current hard-restart segment index.
            local_ids: Predictor-local object ids for this frame.
            masks_this_frame: Binary masks aligned with ``local_ids``.
            reference_bboxes: Ignored; kept for call-site compatibility.

        Returns:
            Video-level global ids aligned with ``local_ids``.
        """
        del reference_bboxes
        local_ids = [_to_int_id(lid) for lid in list(local_ids)]
        masks_this_frame = list(masks_this_frame)
        keys = [(int(segment_idx), int(lid)) for lid in local_ids]
        dtype = np.int64

        if len(keys) == 0:
            return np.array([], dtype=dtype)

        current_valid_lids = set()
        for lid, mask in zip(local_ids, masks_this_frame):
            if self.valid_bbox(mask) is not None:
                current_valid_lids.add(int(lid))

        valid_keys = [
            key
            for lid, key in zip(local_ids, keys)
            if int(lid) in current_valid_lids
        ]
        if all(key in self.id_remap for key in valid_keys):
            return self._result_array(local_ids, keys, current_valid_lids, dtype)

        used_gids = self._used_global_ids()
        for lid, key in zip(local_ids, keys):
            if key in self.id_remap:
                continue
            if int(lid) not in current_valid_lids:
                continue
            self._advance_next_global_id_past(used_gids)
            gid = int(self.next_global_id)
            self.id_remap[key] = gid
            used_gids.add(gid)
            self.next_global_id = gid + 1

        return self._result_array(local_ids, keys, current_valid_lids, dtype)

    def observe(self, gid: int, bbox: object, segment_idx: int) -> None:
        """No-op: ablation mode does not maintain cross-segment lifecycle state."""
        del gid, bbox, segment_idx

    def local_id_to_int(self, local_id) -> int:
        """Normalize a predictor-local id to plain int."""
        return _to_int_id(local_id)

    def get_global_id(self, segment_idx: int, local_id: int) -> Optional[int]:
        """Return the current global id for a segment-local id."""
        gid = self.id_remap.get((int(segment_idx), self.local_id_to_int(local_id)))
        return None if gid is None else int(gid)

    def previous_bboxes(self) -> Dict[int, object]:
        """Return empty refs so callers never cross-link segments by accident."""
        return {}

    def end_segment(self, segment_idx: int, observed_bboxes: Dict[int, object]) -> None:
        """No-op: ablation mode does not persist segment-end geometry."""
        del segment_idx, observed_bboxes
