"""Build tracker instances from YAML config (aligned with demo_video.py)."""

import os

_FLATTEN_SECTIONS = {"io", "tracking"}

_DEFAULTS = {
    "relocation_interval":  20,
    "match_iou_thr":        0.3,
    "max_mask_area_ratio":  0.2,
    "sam3_ckpt":            "checkpoints/sam3.pt",
    "sam2_ckpt":            "checkpoints/sam2.1_hiera_large.pt",
    "yolo_ckpt":            "checkpoints/l_stage1-7d280586.pth",
    "gdino_ckpt":           "checkpoints/groundingdino_swint_ogc.pth",
    "text_threshold":       0.25,
    "use_track_lifecycle":  True,
    "lost_track_ttl_segments": 2,
    "lifecycle_match_score_thr": 0.45,
    "lifecycle_center_gate": 2.5,
    "lifecycle_area_ratio_min": 0.35,
    "lifecycle_area_ratio_max": 2.8,
}


def parse_bool(value):
    """Parse CLI/config booleans without treating arbitrary strings as True."""
    from aerotrack_core.pipeline.track_lifecycle import _coerce_bool

    return _coerce_bool("boolean value", value)


def _validate_tracking_params(args):
    """Normalize tracking params before any model is loaded."""
    from aerotrack_core.pipeline.track_lifecycle import (
        _coerce_bool,
        _float_range,
        _int_range,
    )

    args.match_iou_thr = _float_range(
        "match_iou_thr",
        args.match_iou_thr,
        lower=0.0,
        upper=1.0,
    )
    args.max_mask_area_ratio = _float_range(
        "max_mask_area_ratio",
        args.max_mask_area_ratio,
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
    args.use_track_lifecycle = _coerce_bool(
        "use_track_lifecycle",
        getattr(args, "use_track_lifecycle", True),
    )
    args.lost_track_ttl_segments = _int_range(
        "lost_track_ttl_segments",
        getattr(args, "lost_track_ttl_segments", 2),
        lower=0,
    )
    args.lifecycle_match_score_thr = _float_range(
        "lifecycle_match_score_thr",
        getattr(args, "lifecycle_match_score_thr", 0.45),
        lower=0.0,
        upper=1.0,
    )
    args.lifecycle_center_gate = _float_range(
        "lifecycle_center_gate",
        getattr(args, "lifecycle_center_gate", 2.5),
        lower=0.0,
        lower_inclusive=False,
    )
    args.lifecycle_area_ratio_min = _float_range(
        "lifecycle_area_ratio_min",
        getattr(args, "lifecycle_area_ratio_min", 0.35),
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
    args.lifecycle_area_ratio_max = _float_range(
        "lifecycle_area_ratio_max",
        getattr(args, "lifecycle_area_ratio_max", 2.8),
        lower=1.0,
    )
    if args.lifecycle_area_ratio_max < args.lifecycle_area_ratio_min:
        raise ValueError(
            "lifecycle_area_ratio_max must be >= lifecycle_area_ratio_min"
        )


def load_config(config_path: str) -> dict:
    """Load YAML config and flatten io/tracking sections to top level."""
    import yaml
    with open(config_path, encoding="utf-8-sig") as f:
        raw = yaml.safe_load(f)
    flat = {}
    for key, section in raw.items():
        if isinstance(section, dict) and key in _FLATTEN_SECTIONS:
            flat.update(section)
        else:
            flat[key] = section
    return flat


def build_tracker(args):
    """Build a tracker for the requested baseline (1-9)."""
    for k, v in _DEFAULTS.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    _validate_tracking_params(args)
    tracker_kwargs = dict(
        sam3_checkpoint=args.sam3_ckpt,
        max_objects=args.max_objects,
        new_det_thresh=args.new_det_thresh,
        max_trk_keep_alive=args.max_trk_keep_alive,
        match_iou_thr=args.match_iou_thr,
        max_missing_segments=getattr(args, "max_missing_segments", None) or 1,
        max_mask_area_ratio=args.max_mask_area_ratio,
        verbose=getattr(args, "verbose", False),
        show_progress=False,
    )
    lifecycle_kwargs = dict(
        use_track_lifecycle=getattr(args, "use_track_lifecycle", True),
        lost_track_ttl_segments=getattr(args, "lost_track_ttl_segments", 2),
        lifecycle_match_score_thr=getattr(args, "lifecycle_match_score_thr", 0.45),
        lifecycle_center_gate=getattr(args, "lifecycle_center_gate", 2.5),
        lifecycle_area_ratio_min=getattr(args, "lifecycle_area_ratio_min", 0.35),
        lifecycle_area_ratio_max=getattr(args, "lifecycle_area_ratio_max", 2.8),
    )
    bl = args.baseline

    if bl == 1:
        from aerotrack_core.pipeline.sam3_baseline import SAM3BaselineTracker
        tracker = SAM3BaselineTracker(**tracker_kwargs)

    elif bl == 2:
        from aerotrack_core.pipeline.sam3_reloc_baseline import SAM3RelocTracker
        tracker = SAM3RelocTracker(
            relocation_interval=args.relocation_interval,
            **tracker_kwargs,
            **lifecycle_kwargs,
        )

    elif bl == 3:
        from aerotrack_core.pipeline.yoloworld_baseline import YOLOWorldBaselineTracker
        tracker = YOLOWorldBaselineTracker(
            yolo_checkpoint=args.yolo_ckpt,
            relocation_interval=args.relocation_interval,
            score_thr=args.score_thr,
            max_dets=args.max_dets,
            nms_iou_thr=args.nms_iou_thr,
            sam3_tuning_cfg=getattr(args, "sam3_tuning", None),
            infer_scale=getattr(args, "yolo_infer_scale", 1.0) or 1.0,
            **tracker_kwargs,
            **lifecycle_kwargs,
        )

    elif bl == 4:
        from aerotrack_core.pipeline.gdino_sam3_baseline import GDinoSAM3BaselineTracker
        tracker = GDinoSAM3BaselineTracker(
            gdino_checkpoint=args.gdino_ckpt,
            gdino_bert_path=getattr(args, "gdino_bert_path", None),
            relocation_interval=args.relocation_interval,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_dets=args.max_dets,
            nms_iou_thr=args.nms_iou_thr,
            sam3_tuning_cfg=getattr(args, "sam3_tuning", None),
            infer_scale=getattr(args, "gdino_infer_scale", 1.0) or 1.0,
            **tracker_kwargs,
            **lifecycle_kwargs,
        )

    elif bl in (5, 6):
        sam2_ckpt = getattr(args, "sam2_ckpt", _DEFAULTS["sam2_ckpt"])
        if bl == 5:
            from aerotrack_core.pipeline.yoloworld_sam2_baseline import YOLOWorldSAM2BaselineTracker
            tracker = YOLOWorldSAM2BaselineTracker(
                sam2_checkpoint=sam2_ckpt,
                yolo_checkpoint=args.yolo_ckpt,
                relocation_interval=args.relocation_interval,
                score_thr=args.score_thr,
                max_dets=args.max_dets,
                nms_iou_thr=args.nms_iou_thr,
                match_iou_thr=args.match_iou_thr,
                max_mask_area_ratio=args.max_mask_area_ratio,
                mask_nms_iou_thr=getattr(args, "mask_nms_iou_thr", 0.5),
                max_objects=args.max_objects,
                use_track_lifecycle=getattr(args, "use_track_lifecycle", True),
                lost_track_ttl_segments=getattr(args, "lost_track_ttl_segments", 2),
                lifecycle_match_score_thr=getattr(args, "lifecycle_match_score_thr", 0.45),
                lifecycle_center_gate=getattr(args, "lifecycle_center_gate", 2.5),
                lifecycle_area_ratio_min=getattr(args, "lifecycle_area_ratio_min", 0.35),
                lifecycle_area_ratio_max=getattr(args, "lifecycle_area_ratio_max", 2.8),
                infer_scale=getattr(args, "yolo_infer_scale", 1.0) or 1.0,
                verbose=getattr(args, "verbose", False),
                show_progress=False,
            )
        else:
            from aerotrack_core.pipeline.gdino_sam2_baseline import GDinoSAM2BaselineTracker
            tracker = GDinoSAM2BaselineTracker(
                sam2_checkpoint=sam2_ckpt,
                gdino_checkpoint=args.gdino_ckpt,
                gdino_bert_path=getattr(args, "gdino_bert_path", None),
                relocation_interval=args.relocation_interval,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                max_dets=args.max_dets,
                nms_iou_thr=args.nms_iou_thr,
                match_iou_thr=args.match_iou_thr,
                max_mask_area_ratio=args.max_mask_area_ratio,
                mask_nms_iou_thr=getattr(args, "mask_nms_iou_thr", 0.5),
                max_objects=args.max_objects,
                use_track_lifecycle=getattr(args, "use_track_lifecycle", True),
                lost_track_ttl_segments=getattr(args, "lost_track_ttl_segments", 2),
                lifecycle_match_score_thr=getattr(args, "lifecycle_match_score_thr", 0.45),
                lifecycle_center_gate=getattr(args, "lifecycle_center_gate", 2.5),
                lifecycle_area_ratio_min=getattr(args, "lifecycle_area_ratio_min", 0.35),
                lifecycle_area_ratio_max=getattr(args, "lifecycle_area_ratio_max", 2.8),
                infer_scale=getattr(args, "gdino_infer_scale", 1.0) or 1.0,
                verbose=getattr(args, "verbose", False),
                show_progress=False,
            )
        tracker.load_models()
        return tracker

    elif bl == 7:
        from aerotrack_core.pipeline.sam3_incremental_baseline import SAM3IncrementalTracker
        tracker = SAM3IncrementalTracker(
            relocation_interval=args.relocation_interval,
            **tracker_kwargs,
        )

    elif bl == 8:
        from aerotrack_core.pipeline.gdino_incremental_baseline import GDinoIncrementalTracker
        tracker = GDinoIncrementalTracker(
            gdino_checkpoint=args.gdino_ckpt,
            gdino_bert_path=getattr(args, "gdino_bert_path", None),
            relocation_interval=args.relocation_interval,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_dets=args.max_dets,
            nms_iou_thr=args.nms_iou_thr,
            sam3_tuning_cfg=getattr(args, "sam3_tuning", None),
            **tracker_kwargs,
        )

    elif bl == 9:
        from aerotrack_core.pipeline.yoloworld_incremental_baseline import YOLOWorldIncrementalTracker
        tracker = YOLOWorldIncrementalTracker(
            yolo_checkpoint=args.yolo_ckpt,
            relocation_interval=args.relocation_interval,
            score_thr=args.score_thr,
            max_dets=args.max_dets,
            nms_iou_thr=args.nms_iou_thr,
            sam3_tuning_cfg=getattr(args, "sam3_tuning", None),
            **tracker_kwargs,
        )

    else:
        raise ValueError(f"Unknown baseline: {bl}. Valid choices: 1-9.")

    tracker.load_models()

    # Apply sam3_tuning memory-bank overrides after model load (idempotent).
    sam3_tuning = getattr(args, "sam3_tuning", None) or {}
    if sam3_tuning:
        model = getattr(tracker, "video_predictor", None)
        model = getattr(model, "model", None) if model else None
        trk = getattr(model, "tracker", None) if model else None
        if trk is not None:
            if sam3_tuning.get("num_maskmem") is not None:
                trk.num_maskmem = int(sam3_tuning["num_maskmem"])
            if sam3_tuning.get("max_obj_ptrs_in_encoder") is not None:
                trk.max_obj_ptrs_in_encoder = int(sam3_tuning["max_obj_ptrs_in_encoder"])

    return tracker
