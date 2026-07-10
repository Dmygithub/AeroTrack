"""Lifecycle-aware local-to-global ID association.

This module is the glue around ``TrackLifecycleBank``.  It maps segment-local
object ids produced by SAM3/SAM2 to video-level global ids, filters degenerated
masks, and handles local-id handover inside a segment.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from aerotrack_core.pipeline.track_lifecycle import (
    TrackLifecycleBank,
    _float_range,
    _int_range,
)


def mask_to_bbox(mask) -> Optional[List[int]]:
    """Return [x1, y1, x2, y2] from a 2-D/1xHxW mask, or None if empty."""
    if mask is None:
        return None
    m = mask[0] if getattr(mask, "ndim", 0) == 3 else mask
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _to_int_id(value) -> int:
    """Convert Python/numpy/torch scalar ids to plain int."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def build_video_id_association(tracker, *, frame_area: int):
    """Return the ID association implementation for hard-restart trackers.

    When ``use_track_lifecycle`` is enabled, returns ``LifecycleIDAssociation``
    (cross-segment LIA).  When disabled, returns ``SegmentIsolatedIDAssociation``
    for w/o LIA ablation without altering the enabled code path.

    Args:
        tracker: Tracker instance providing lifecycle-related attributes.
        frame_area: Product of frame height and width in pixels.

    Returns:
        An object exposing ``assign_frame``, ``valid_bbox``, ``observe``,
        ``end_segment``, ``previous_bboxes``, and ``local_id_to_int``.
    """
    if getattr(tracker, "use_track_lifecycle", True):
        return LifecycleIDAssociation.from_tracker(tracker, frame_area=frame_area)

    from aerotrack_core.pipeline.segment_isolated_id_association import (
        SegmentIsolatedIDAssociation,
    )

    return SegmentIsolatedIDAssociation.from_tracker(tracker, frame_area=frame_area)


class LifecycleIDAssociation:
    """Maintain video-level ids for hard-restart tracking segments."""

    def __init__(
        self,
        *,
        max_mask_area_ratio: float = 0.5,
        frame_area: int,
        match_score_thr: float = 0.45,
        center_gate: float = 2.5,
        area_ratio_min: float = 0.35,
        area_ratio_max: float = 2.8,
        lost_ttl_segments: int = 2,
    ):
        self.max_mask_area_ratio = _float_range(
            "max_mask_area_ratio",
            max_mask_area_ratio,
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        self.frame_area = _int_range("frame_area", frame_area, lower=1)
        self.lifecycle = TrackLifecycleBank(
            enabled=True,
            match_score_thr=match_score_thr,
            center_gate=center_gate,
            area_ratio_min=area_ratio_min,
            area_ratio_max=area_ratio_max,
            lost_ttl_segments=lost_ttl_segments,
        )
        self.id_remap: Dict[Tuple[int, int], int] = {}
        self.next_global_id = 0
        self.prev_segment_bboxes: Dict[int, object] = {}

    def _used_global_ids(self) -> set:
        """Return all global ids known to LIA and the lifecycle bank."""
        used = {int(gid) for gid in self.id_remap.values()}
        used.update(int(gid) for gid in getattr(self.lifecycle, "active", {}).keys())
        used.update(int(gid) for gid in getattr(self.lifecycle, "lost", {}).keys())
        return used

    def _advance_next_global_id_past(self, used_gids: set) -> None:
        """Keep newly allocated ids from colliding with existing/seeded ids."""
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

    @classmethod
    def from_tracker(cls, tracker, *, frame_area: int) -> "LifecycleIDAssociation":
        return cls(
            max_mask_area_ratio=getattr(tracker, "max_mask_area_ratio", 0.5),
            frame_area=frame_area,
            match_score_thr=getattr(tracker, "lifecycle_match_score_thr", 0.45),
            center_gate=getattr(tracker, "lifecycle_center_gate", 2.5),
            area_ratio_min=getattr(tracker, "lifecycle_area_ratio_min", 0.35),
            area_ratio_max=getattr(tracker, "lifecycle_area_ratio_max", 2.8),
            lost_ttl_segments=getattr(tracker, "lost_track_ttl_segments", 2),
        )

    @property
    def enabled(self) -> bool:
        return True

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

    def assign_frame(
        self,
        segment_idx: int,
        local_ids: Iterable[int],
        masks_this_frame: Iterable[object],
        *,
        reference_bboxes: Optional[Dict[int, object]] = None,
    ) -> np.ndarray:
        """Assign frame-local object ids to global ids via LIA.

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

        cur_bboxes = {}
        current_valid_lids = set()
        for lid, mask in zip(local_ids, masks_this_frame):
            lid = int(lid)
            bbox = self.valid_bbox(mask)
            cur_bboxes[lid] = bbox
            if bbox is not None:
                current_valid_lids.add(lid)

        valid_keys = [
            key
            for lid, key in zip(local_ids, keys)
            if int(lid) in current_valid_lids
        ]
        if all(key in self.id_remap for key in valid_keys):
            return self._result_array(local_ids, keys, current_valid_lids, dtype)

        unresolved = {
            int(lid): cur_bboxes.get(int(lid))
            for lid, key in zip(local_ids, keys)
            if int(lid) in current_valid_lids and key not in self.id_remap
        }
        forbidden_gids = [
            int(self.id_remap[key])
            for lid, key in zip(local_ids, keys)
            if int(lid) in current_valid_lids and key in self.id_remap
        ]
        self._advance_next_global_id_past(self._used_global_ids())
        assigned, next_id = self.lifecycle.assign(
            unresolved,
            next_global_id=self.next_global_id,
            segment_idx=int(segment_idx),
            forbidden_gids=forbidden_gids,
        )
        max_assigned_gid = max((int(gid) for gid in assigned.values()), default=-1)
        self.next_global_id = max(int(next_id), max_assigned_gid + 1)
        for lid, gid in assigned.items():
            self._handover_segment_gid(
                segment_idx=int(segment_idx),
                new_lid=int(lid),
                gid=int(gid),
                current_valid_lids=current_valid_lids,
            )
            self.id_remap[(int(segment_idx), int(lid))] = int(gid)
        return self._result_array(local_ids, keys, current_valid_lids, dtype)

    def _handover_segment_gid(
        self,
        *,
        segment_idx: int,
        new_lid: int,
        gid: int,
        current_valid_lids: set,
    ) -> None:
        """Allow a new local id to inherit a gid from an invalid old local id."""
        for old_key, old_gid in list(self.id_remap.items()):
            if (
                int(old_key[0]) == int(segment_idx)
                and int(old_gid) == int(gid)
                and int(old_key[1]) != int(new_lid)
                and int(old_key[1]) not in current_valid_lids
            ):
                del self.id_remap[old_key]

    def observe(self, gid: int, bbox: object, segment_idx: int) -> None:
        self.lifecycle.observe(int(gid), bbox, int(segment_idx))

    def local_id_to_int(self, local_id) -> int:
        """Normalize a predictor-local id to plain int."""
        return _to_int_id(local_id)

    def get_global_id(self, segment_idx: int, local_id: int) -> Optional[int]:
        """Return the current global id for a segment-local id."""
        gid = self.id_remap.get((int(segment_idx), self.local_id_to_int(local_id)))
        return None if gid is None else int(gid)

    def previous_bboxes(self) -> Dict[int, object]:
        """Return last segment's final valid bboxes."""
        return dict(self.prev_segment_bboxes)

    def end_segment(self, segment_idx: int, observed_bboxes: Dict[int, object]) -> None:
        self.prev_segment_bboxes = dict(observed_bboxes or {})
        self.lifecycle.end_segment(int(segment_idx), self.prev_segment_bboxes)
