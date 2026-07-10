"""Conservative cross-segment track lifecycle matching.

This module keeps the ID assignment logic shared by hard-restart baselines
small and detector-agnostic.  It does not change detection or mask propagation;
it only decides whether a local object in the current segment should reuse an
older global track id.
"""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional, Tuple


BBox = Tuple[float, float, float, float]


def _coerce_bool(name: str, value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean value")


def _finite_float(name: str, value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _float_range(
    name: str,
    value,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> float:
    value = _finite_float(name, value)
    if lower is not None:
        if lower_inclusive and value < lower:
            raise ValueError(f"{name} must be >= {lower}")
        if not lower_inclusive and value <= lower:
            raise ValueError(f"{name} must be > {lower}")
    if upper is not None:
        if upper_inclusive and value > upper:
            raise ValueError(f"{name} must be <= {upper}")
        if not upper_inclusive and value >= upper:
            raise ValueError(f"{name} must be < {upper}")
    return value


def _int_range(name: str, value, *, lower: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if int_value != value and not isinstance(value, str):
        raise ValueError(f"{name} must be an integer")
    if lower is not None and int_value < lower:
        raise ValueError(f"{name} must be >= {lower}")
    return int_value


@dataclass
class TrackState:
    gid: int
    bbox: BBox
    last_segment: int
    missed_segments: int = 0


def _as_bbox(box) -> Optional[BBox]:
    if box is None:
        return None
    try:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_iou(a, b) -> float:
    a = _as_bbox(a)
    b = _as_bbox(b)
    if a is None or b is None:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _center(box: BBox) -> Tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


class TrackLifecycleBank:
    """Active/lost track bank for hard-restart baselines.

    Matching uses a conservative geometry score:
      0.45 * bbox IoU + 0.30 * center similarity + 0.25 * area similarity.
    Lost tracks receive a small age penalty so a recent active track wins over
    an older disappeared track when both are plausible.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        match_score_thr: float = 0.45,
        center_gate: float = 2.5,
        area_ratio_min: float = 0.35,
        area_ratio_max: float = 2.8,
        lost_ttl_segments: int = 2,
    ):
        self.enabled = _coerce_bool("use_track_lifecycle", enabled)
        self.match_score_thr = _float_range(
            "lifecycle_match_score_thr",
            match_score_thr,
            lower=0.0,
            upper=1.0,
        )
        self.center_gate = _float_range(
            "lifecycle_center_gate",
            center_gate,
            lower=0.0,
            lower_inclusive=False,
        )
        self.area_ratio_min = _float_range(
            "lifecycle_area_ratio_min",
            area_ratio_min,
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        self.area_ratio_max = _float_range(
            "lifecycle_area_ratio_max",
            area_ratio_max,
            lower=1.0,
        )
        if self.area_ratio_max < self.area_ratio_min:
            raise ValueError(
                "lifecycle_area_ratio_max must be >= lifecycle_area_ratio_min"
            )
        self.lost_ttl_segments = _int_range(
            "lost_track_ttl_segments",
            lost_ttl_segments,
            lower=0,
        )
        self.active: Dict[int, TrackState] = {}
        self.lost: Dict[int, TrackState] = {}

    def reset(self) -> None:
        self.active.clear()
        self.lost.clear()

    def _score(self, cur: BBox, state: TrackState, source: str) -> Optional[float]:
        ref = _as_bbox(state.bbox)
        if ref is None:
            return None
        cur_area = _area(cur)
        ref_area = _area(ref)
        if cur_area <= 0.0 or ref_area <= 0.0:
            return None

        rel_area = cur_area / ref_area
        if rel_area < self.area_ratio_min or rel_area > self.area_ratio_max:
            return None

        cx, cy = _center(cur)
        rx, ry = _center(ref)
        dist = math.hypot(cx - rx, cy - ry)
        max_dist = max(1.0, self.center_gate * math.sqrt(max(cur_area, ref_area)))
        if dist > max_dist:
            return None

        iou = bbox_iou(cur, ref)
        center_sim = max(0.0, 1.0 - dist / max_dist)
        area_sim = min(cur_area, ref_area) / max(cur_area, ref_area)
        score = 0.45 * iou + 0.30 * center_sim + 0.25 * area_sim
        if source == "lost":
            score -= 0.03 * max(1, int(state.missed_segments))
        return float(score)

    def assign(
        self,
        local_bboxes: Dict[int, object],
        *,
        next_global_id: int,
        segment_idx: int,
        forbidden_gids: Optional[Iterable[int]] = None,
    ) -> Tuple[Dict[int, int], int]:
        """Assign local ids to global ids.

        Args:
            local_bboxes: Mapping local object id -> bbox xyxy.
            next_global_id: First unused global id.
            segment_idx: Current segment index.

        Returns:
            (local_to_global, next_global_id_after_new_allocations)
        """
        local_ids = [int(lid) for lid in local_bboxes.keys()]
        blocked_gids = {int(gid) for gid in (forbidden_gids or [])}
        known_gids = set(blocked_gids)
        known_gids.update(int(gid) for gid in self.active.keys())
        known_gids.update(int(gid) for gid in self.lost.keys())

        def allocate_new_gid() -> int:
            nonlocal next_global_id
            while int(next_global_id) in known_gids:
                next_global_id += 1
            gid = int(next_global_id)
            known_gids.add(gid)
            next_global_id = gid + 1
            return gid

        if not self.enabled:
            assignments: Dict[int, int] = {}
            for lid in local_ids:
                gid = allocate_new_gid()
                assignments[lid] = gid
            return assignments, int(next_global_id)

        valid = {
            int(lid): box
            for lid, raw_box in local_bboxes.items()
            for box in [_as_bbox(raw_box)]
            if box is not None
        }
        assignments: Dict[int, int] = {}
        candidates = []

        for lid, box in valid.items():
            for gid, state in self.active.items():
                if int(gid) in blocked_gids:
                    continue
                score = self._score(box, state, "active")
                if score is not None and score >= self.match_score_thr:
                    candidates.append((score + 0.02, lid, gid, "active"))
            for gid, state in self.lost.items():
                if int(gid) in blocked_gids:
                    continue
                if state.missed_segments > self.lost_ttl_segments:
                    continue
                score = self._score(box, state, "lost")
                if score is not None and score >= self.match_score_thr:
                    candidates.append((score, lid, gid, "lost"))

        candidates.sort(reverse=True)
        for _score, lid, gid, source in candidates:
            if lid in assignments or gid in blocked_gids:
                continue
            assignments[lid] = gid
            blocked_gids.add(gid)
            box = valid[lid]
            if source == "lost":
                self.lost.pop(gid, None)
            self.active[gid] = TrackState(
                gid=gid,
                bbox=box,
                last_segment=segment_idx,
                missed_segments=0,
            )

        for lid, box in valid.items():
            if lid in assignments:
                continue
            gid = allocate_new_gid()
            assignments[lid] = gid
            self.active[gid] = TrackState(
                gid=gid,
                bbox=box,
                last_segment=segment_idx,
                missed_segments=0,
            )

        for lid in local_ids:
            if lid in assignments:
                continue
            gid = allocate_new_gid()
            assignments[lid] = gid

        return assignments, int(next_global_id)

    def end_segment(self, segment_idx: int, observed_bboxes: Dict[int, object]) -> None:
        """Update active/lost status from the final valid boxes of a segment."""
        if not self.enabled:
            return

        observed = {
            int(gid): box
            for gid, raw_box in observed_bboxes.items()
            for box in [_as_bbox(raw_box)]
            if box is not None
        }

        for gid, box in observed.items():
            self.lost.pop(gid, None)
            self.active[gid] = TrackState(
                gid=gid,
                bbox=box,
                last_segment=segment_idx,
                missed_segments=0,
            )

        for gid in list(self.active.keys()):
            if gid in observed:
                continue
            state = self.active.pop(gid)
            state.missed_segments += 1
            state.last_segment = segment_idx
            if state.missed_segments <= self.lost_ttl_segments:
                self.lost[gid] = state

        for gid in list(self.lost.keys()):
            if gid in observed:
                continue
            state = self.lost[gid]
            # Tracks moved from active to lost above were already aged once for
            # this segment; do not age them twice.
            if state.last_segment == segment_idx:
                continue
            state.missed_segments += 1
            state.last_segment = segment_idx
            if state.missed_segments > self.lost_ttl_segments:
                del self.lost[gid]

    def observe(self, gid: int, bbox: object, segment_idx: int) -> None:
        """Refresh a track with a valid in-segment observation.

        This does not age unobserved tracks. It only keeps active geometry
        current so a target that briefly receives a new local obj_id inside the
        same segment can still be matched to the latest known position.
        """
        if not self.enabled:
            return
        box = _as_bbox(bbox)
        if box is None:
            return
        gid = int(gid)
        self.lost.pop(gid, None)
        self.active[gid] = TrackState(
            gid=gid,
            bbox=box,
            last_segment=segment_idx,
            missed_segments=0,
        )

    def seed_from_previous(self, bboxes: Dict[int, object], segment_idx: int) -> None:
        """Compatibility helper for starting from an existing gid->bbox map."""
        for gid, box in bboxes.items():
            box = _as_bbox(box)
            if box is None:
                continue
            gid = int(gid)
            self.active[gid] = TrackState(
                gid=gid,
                bbox=box,
                last_segment=segment_idx,
                missed_segments=0,
            )
